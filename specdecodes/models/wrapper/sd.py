import json
import logging
import os
import random
import time
import torch
import torch.nn.functional as F
from .base import WrapperBase

from transformers.generation.logits_process import LogitsWarper
from transformers.generation.stopping_criteria import StoppingCriteria

from bigtree import preorder_iter, levelorder_iter, shift_nodes, find_attrs
from bigtree import tree_to_nested_dict
from ..utils import TreeDynamicCache, build_tree_attention_data, get_residual


class SDWrapper(WrapperBase):
    def __init__(self, method="naive"):
        super(SDWrapper, self).__init__()
        self.method = method
  
    def set_ssm(self, ssm):
        self.ssm = ssm
    
    def _speculate(self, hidden_states, input_ids, logits_warper, do_sample, past_key_values, eos_token_id=None):
        return self.ssm.speculate(
            hidden_states,
            input_ids,
            embed_tokens=self.llm.get_input_embeddings(), 
            lm_head=self.llm.lm_head,
            logits_warper=logits_warper,
            do_sample=do_sample,
            past_key_values=past_key_values,
            eos_token_id=eos_token_id,
        )
        
    def _tree_decoding(self, root, past_key_values, position_offset, device, dtype=torch.float32):
        # Preparing llm's tree decoding data, also updates each node's index (node.ind).
        tree_input_ids, tree_position_ids, tree_mask = build_tree_attention_data(root, position_offset=position_offset, dtype=dtype)
        
        # Move to device
        tree_input_ids = tree_input_ids.to(device)
        tree_position_ids = tree_position_ids.to(device)
        tree_mask = tree_mask.to(device)
        
        # llm forward
        outputs = self.llm(
            tree_input_ids,
            past_key_values=past_key_values,
            attention_mask=tree_mask,
            position_ids=tree_position_ids,
            output_hidden_states=True,
        )
        
        return outputs
    
    #TODO: Implement stochastic method and ensure correctness
    def _verify(self, root, logits, logits_warper, do_sample, eos_token_id=None, sampling_method="naive"):
        sampled_tokens = []
        hidden_indices = []
        if do_sample == False:
            logging.debug("'do_sample' is False, sampling_method will be set to 'naive'.")
            sampling_method = "naive"
            
        if sampling_method == "naive":
            real_token_ids = self._sample_token(logits, logits_warper, do_sample=do_sample).squeeze(0) # remove batch dim

            # for each depth, find token that matches the predicted token
            # if all tokens are matched, then the last token is the bonus token
            cur = root
            while cur.children:
                llm_token_id = real_token_ids[cur.ind]
                
                accepts_token = False
                for child in cur.children:
                    if child.id == llm_token_id:
                        accepts_token = True
                        sampled_tokens.append(child.id) # real_token_ids[cur.ind]
                        hidden_indices.append(cur.ind)
                        cur = child
                        break
                    
                # stop loop if no token is accepted
                if accepts_token == False: break
            
            if len(sampled_tokens) == 0 or sampled_tokens[-1] != eos_token_id: # no token matched, accept first token
                bonus_token = real_token_ids[cur.ind].item()
                sampled_tokens.append(bonus_token)
                hidden_indices.append(cur.ind)

        elif sampling_method == "fast": # should be equivalent to eagle method
            assert do_sample == True, "Fast method requires sampling"
            global_p = self._sample_token(logits, logits_warper, do_sample=do_sample, return_probs=True).squeeze(0) # remove batch dim
            
            cur = root
            while cur.children:
                sampled_id = global_p[cur.ind].multinomial(num_samples=1).item()
                child_ids = [child.id for child in cur.children]
                child_id_to_node = {node.id: node for node in cur.children}
                if sampled_id in child_ids:
                    hidden_indices.append(cur.ind)
                    sampled_tokens.append(sampled_id)
                    cur = cur.children[child_id_to_node[sampled_id]]
                else:
                    for child_id in child_ids:
                        global_p[cur.ind][child_id] = 0
                    global_p[cur.ind] = global_p[cur.ind] / global_p[cur.ind].sum()
                    break
            
            # generate bonus token, don't generate if eos token is already the last token
            if len(sampled_tokens) == 0 or sampled_tokens[-1] != eos_token_id:
                bonus_token = global_p[cur.ind].multinomial(num_samples=1).item()
                sampled_tokens.append(bonus_token)
                hidden_indices.append(cur.ind)
                
        elif sampling_method == "eagle":
            assert do_sample == True, "Eagle method requires sampling"
            global_p = self._sample_token(logits, logits_warper, do_sample=do_sample, return_probs=True).squeeze(0) # remove batch dim
            
            cur = root
            while cur.children:
                
                accepts_token = False
                for child in cur.children:
                    r = random.random()
                    # qx = gtp[child.id]
                    # px = 1  # in this iteration, only child.id' prob is 1, others are 0.
                    # if r <= qx / px:
                    if r <= global_p[cur.ind][child.id]: # since px = 1, we can compare r with qx directly
                        accepts_token = True
                        sampled_tokens.append(child.id)
                        hidden_indices.append(cur.ind)
                        cur = child
                        break
                    else:
                        # global_p[cur.ind] = torch.clamp(global_p[cur.ind] - p, min=0)
                        global_p[cur.ind][child.id] = 0 # only child.id' prob is 1, equivalent to function above
                        global_p[cur.ind] = global_p[cur.ind] / global_p[cur.ind].sum()
                        # p[child.id] = 0
                        # p = p / p.sum()
                        
                # stop loop if no token is accepted
                if accepts_token == False: break
                
            if len(sampled_tokens) == 0 or sampled_tokens[-1] != eos_token_id: # eos token should be the last token
                bonus_token = global_p[cur.ind].multinomial(num_samples=1).item()
                sampled_tokens.append(bonus_token)
                hidden_indices.append(cur.ind)
        
        elif sampling_method == "sequoia":
            assert do_sample == True, "real_sequoia method requires sampling"
            global_p = self._sample_token(logits, logits_warper, do_sample=do_sample, return_probs=True).squeeze(0) # remove batch dim
            
            cur = root
            while cur.children:
                q = cur.sample_probs
                # keep the non-repacement sampling order
                children = [ child for child in cur.children ]
                children.sort(key=lambda x: x.order)
                
                tried_ids = []
                accepts_token = False
                for child in children:
                    r = random.random()
                    if r <= global_p[cur.ind][child.id] / q[child.id]:
                        accepts_token = True
                        sampled_tokens.append(child.id)
                        hidden_indices.append(cur.ind)
                        cur = child
                        break
                    else:
                        global_p[cur.ind] = get_residual(global_p[cur.ind], q)
                        q[child.id] = 0
                        
                        # Below only benefits when sampled ids have prob 0, which is rarely the case.
                        # set q[t] = 0 if t in S, else 1
                        tried_ids.append(cur.ind)
                        if q.sum() == 0:
                            child_ids = [child.id for child in cur.children]
                            q = torch.zeros_like(q)
                            q[child_ids] = 1
                            q[tried_ids] = 0
                        
                        q = q / q.sum()
                        
                # stop loop if no token is accepted
                if accepts_token == False: break
             
            if len(sampled_tokens) == 0 or sampled_tokens[-1] != eos_token_id: # eos token should be the last token
                bonus_token = global_p[cur.ind].multinomial(num_samples=1).item()
                sampled_tokens.append(bonus_token)
                hidden_indices.append(cur.ind)
        
        
        elif sampling_method == "accept1":
            real_token_ids = self._sample_token(logits[:, :1, :], logits_warper, do_sample=do_sample).squeeze(0) # remove batch dim

            cur = root
            if len(sampled_tokens) == 0 or sampled_tokens[-1] != eos_token_id: # no token matched, accept first token
                bonus_token = real_token_ids[cur.ind].item()
                sampled_tokens.append(bonus_token)
                hidden_indices.append(cur.ind)
        
        else:
            raise ValueError("Invalid method")
        
        # Convert the sampled tokens and hidden indices to tensors
        sampled_tokens = torch.tensor(sampled_tokens, dtype=torch.long)[None] # add back batch size dim
        hidden_indices = torch.tensor(hidden_indices, dtype=torch.long)
        
        return sampled_tokens, hidden_indices

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_warper: LogitsWarper,
        do_sample: bool,
    ):
        """
        Generate sequence of tokens with speculative decoding.

        This method consists of two main stages: prefill and decode.

        Prefill Stage:
        - Perform the model's initial forward pass.
        - Sample a token and append it to the input_ids.

        Decode Stage (with speculative decoding):
        - Iterate through the following steps:
            1. Perform SSM speculative sampling, returns sampled tokens in tree form.
            2. Decode the sampled tokens in parallel with the language model (LLM), generating probabilities for each token.
            3. Verify the sampled tokens by accepting or rejecting them, corresponding to the probabilities.
            4. Update the key-value cache, input_ids, and hidden_states accordingly.

        Args:
            input_ids (torch.LongTensor): The input token IDs. 
            stopping_criteria (StoppingCriteria): The criteria to stop the generation.
            logits_warper (LogitsWarper): The warper to modify the logits.
            do_sample (bool): Whether to sample tokens during generation. If False, the generation will be deterministic.

        Returns:
            input_ids (torch.LongTensor): The generated token IDs.
        """
        assert self.llm is not None, "LLM model must be provided"
        assert self.ssm is not None, "SSM model must be provided"
        assert self.tokenizer is not None, "Tokenizer must be provided"

        # * clone input_ids 
        input_ids = input_ids.clone()

        # * prepare kv-cache
        llm_past_key_values = TreeDynamicCache()
        ssm_past_key_values = TreeDynamicCache()

        # * prefill stage
        outputs = self.llm(input_ids, past_key_values=llm_past_key_values, output_hidden_states=True)
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1:, :].clone() # hf uses outputs.logits[:, -1, :] instead
        hidden_states = outputs.hidden_states[-1].clone()

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs
        
        next_tokens = self._sample_token(next_token_logits, logits_warper, do_sample)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        finished = False
        while not finished:
            # * speculate
            root = self._speculate(hidden_states, input_ids, logits_warper, do_sample, ssm_past_key_values, eos_token_id=self.tokenizer.eos_token_id)

            # * tree decoding
            prev_kv_len = llm_past_key_values.get_seq_length()
            outputs = self._tree_decoding(root, llm_past_key_values, position_offset=input_ids.shape[1]-1, device=hidden_states.device, dtype=hidden_states.dtype)
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits
            hidden_states = outputs.hidden_states[-1].clone()
            
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

            # * verify
            sampled_tokens, hidden_indices = self._verify(
                                                root, next_token_logits, 
                                                logits_warper, 
                                                do_sample,
                                                eos_token_id=self.tokenizer.eos_token_id,
                                                sampling_method=self.method,
                                            )
            
            sampled_tokens = sampled_tokens.to(input_ids.device)
            hidden_indices = hidden_indices.to(hidden_states.device)

            # * update input_ids, hidden_states, and kv-cache
            # llm_past_key_values.crop(prev_kv_len)
            input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
            hidden_states = hidden_states[:, hidden_indices].clone()
            llm_past_key_values.reorder_cache_with_offset(hidden_indices, offset=prev_kv_len, dim=2)
            
            # * check stopping criteria
            finished = stopping_criteria(input_ids, None)
        
        return input_ids
    

class ProfileSDWrapper(SDWrapper):
    def __init__(self, method="naive", out_dir="specdecodes/experiments/profile_data", prefix="sd"):
        super(ProfileSDWrapper, self).__init__(method)
        self.profile_data = {}
        self.sampled_count = 1 # assume first token is sampled (prefill stage)
        self.iter_count = 1 # assume first step is done (prefill stage)
        
        self.out_dir = out_dir
        self.prefix = prefix
        
    
    def _verify(self, root, logits, logits_warper, do_sample, eos_token_id=None, sampling_method="naive"):
        sampled_tokens, hidden_indices = super(ProfileSDWrapper, self)._verify(root, logits, logits_warper, do_sample, eos_token_id, sampling_method)
        
        # tokenize ids
        nodes = list(preorder_iter(root))
        for node in nodes:
            node.id = self.tokenizer.decode(torch.tensor([node.id]), clean_up_tokenization_spaces=False)
        
        # to compute TVD between p and q
        # tvd = 0.5 * torch.sum(torch.abs(p - q))
        
        # profile data
        # json_graph = tree_to_nested_dict(root, name_key="name", attr_dict={"id": "id", "prob": "prob", "global_prob": "global_prob"})
        # sampled_tokens_list = sampled_tokens.squeeze(0).tolist()
        # self.profile_data[self.iter_count] = {}
        # self.profile_data[self.iter_count]["draft_tree"] = json_graph
        # self.profile_data[self.iter_count]["sampled_tokens"] = sampled_tokens_list
        if self.profile_data.get('iter') is None:
            self.profile_data['iter'] = []
            
        sampled_tokens_list = sampled_tokens.squeeze(0).tolist()
        self.profile_data['iter'].append(sampled_tokens_list)
        
        # logging
        logging.debug(
            f"Total: {len(list(preorder_iter(root)))},"\
            f"\tPredicted: {self.tokenizer.batch_decode(sampled_tokens.squeeze(0), clean_up_tokenization_spaces=False)}"
        )
        
        # update stats
        self.sampled_count += len(sampled_tokens[0])
        self.iter_count += 1
        
        return sampled_tokens, hidden_indices
    
    
    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_warper: LogitsWarper,
        do_sample: bool,
    ):
        cur_time = time.strftime("%Y%m%d-%H%M%S")
        
        # prepare output directory
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            out_path = os.path.join(self.out_dir, f"{self.prefix}_{cur_time}.json")
        else:
            out_path = None
        
        # run generation
        input_ids = super(ProfileSDWrapper, self)._generate(input_ids, stopping_criteria, logits_warper, do_sample)
        
        # logging
        total_sampled = self.sampled_count
        total_iterations = self.iter_count
        avg_sampled = total_sampled / total_iterations
        logging.info(
            f"Total sampled: {total_sampled},"\
            f"\tTotal iterations: {total_iterations},"\
            f"\tAverage sampled: {avg_sampled:.2f}"
        )
        
        # save profile data
        self.profile_data["total_sampled"] = total_sampled
        self.profile_data["total_iterations"] = total_iterations
        self.profile_data["average_sampled"] = avg_sampled
        if self.out_dir is not None:
            with open(out_path, "w") as f:
                json.dump(self.profile_data, f)
        
        return input_ids