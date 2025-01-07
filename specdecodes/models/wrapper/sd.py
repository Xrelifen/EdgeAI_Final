import json
import logging
import os
import time
import numpy as np
import torch
from transformers.generation.logits_process import LogitsWarper
from transformers.generation.stopping_criteria import StoppingCriteria
import prettytable as pt
from .base import WrapperBase
from ..utils import DraftParams

import nvtx

class SDWrapper(WrapperBase):
    def __init__(self, draft_params: DraftParams = None, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        self.draft_params = DraftParams() if draft_params is None else draft_params
    
    def set_ssm(self, ssm):
        self.ssm = ssm
        self.ssm.draft_params = self.draft_params
    
    def _speculate(self, input_ids, hidden_states, past_key_values, max_cache_len=None):
        # if self.ssm.lm_head has attribute, use it, otherwise use llm's lm_head
        if hasattr(self.ssm, "lm_head"):
            lm_head = self.ssm.lm_head
        else:
            lm_head = self.llm.lm_head
            
        return self.ssm.speculate(
            input_ids,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            embed_tokens=self.llm.get_input_embeddings(), 
            lm_head=lm_head,
            max_cache_len=max_cache_len,
        )

    def _tree_decoding(self, tree, tree_mask_base, past_key_values, position_offset, cache_position, device):
        # Preparing llm's tree decoding data, also updates each node's index (node.ind).
        node_data = tree.get_node_data()
        tree_input_ids = node_data['token_ids']
        tree_position_ids = node_data['depths'] + position_offset
        tree_mask = tree.create_attention_mask(position_offset)
        
        # Move to device
        tree_input_ids = tree_input_ids.to(device, non_blocking=True)
        tree_position_ids = tree_position_ids.to(device, non_blocking=True)
        tree_mask = tree_mask.to(device, non_blocking=True)
        
        # Assing to tree mask
        tree_mask_base[:, :, :, :tree_mask.shape[3]] = tree_mask
        
        #invert
        tree_mask = (~tree_mask_base).to(torch.float16) * torch.finfo(torch.float16).min
        
        # llm forward
        #TODO: Remove unnecessary squeeze(0) and unsqueeze(0) operations
        outputs = self.llm(
            tree_input_ids.unsqueeze(0),
            past_key_values=past_key_values,
            attention_mask=tree_mask,
            position_ids=tree_position_ids.unsqueeze(0),
            output_hidden_states=True,
            cache_position=cache_position
        )
        return outputs
    
    def _verify_step(self, p, q, token_ids, do_sample):
        sampled_token_id = p.argmax() if not do_sample else p.multinomial(1).squeeze(-1)
        if torch.any(sampled_token_id == token_ids):
            return sampled_token_id, None
        
        denom = 1.0 - p[token_ids].sum()
        p.div_(denom) if denom >= 1e-9 else p.zero_() # numerical stability
        p[token_ids].zero_()
        return None, p

    def _verify(self, tree, logits, logits_warper, do_sample):
        def sample_token_method(logits, return_probs=False):
            return self._sample_token(logits, logits_warper, do_sample=do_sample, return_probs=return_probs)
        
        # Obtain LLM sample logits
        global_p = sample_token_method(logits, return_probs=True).squeeze(0).to(device='cpu', non_blocking=True) # remove batch dim
        
        # Initialize variables
        sampled_tokens = torch.tensor([], dtype=torch.long, device='cpu')
        hidden_indices = torch.tensor([], dtype=torch.long, device='cpu')
        total_len = 0
        accept_len = 0
        
        # Iterate through draft tree, verify each node
        node_data = tree.get_node_data()
        token_ids = node_data['token_ids']
        token_probs = node_data['cumulative_probabilities']
        
        cur_ind = torch.tensor([0], dtype=torch.long, device='cpu')
        children_inds = tree.get_children_indices(cur_ind)
        children_token_ids = token_ids[children_inds]
        
        torch.cuda.synchronize() # synchronize before starting the loop
        while children_inds.size(0) > 0:
            total_len += 1
            #TODO: Remove unnecessary squeeze(0) and unsqueeze(0) operations
            accept_token_id, new_p = self._verify_step(global_p[cur_ind].squeeze(0), token_probs[cur_ind].squeeze(0), children_token_ids, do_sample)
                    
            # Accept token if it is in the children
            if accept_token_id is not None:
                accept_len += 1
                sampled_tokens = torch.cat([sampled_tokens, accept_token_id[None]])
                hidden_indices = torch.cat([hidden_indices, cur_ind])
                cur_ind = children_inds[children_token_ids == accept_token_id]
                children_inds = tree.get_children_indices(cur_ind)
                children_token_ids = token_ids[children_inds]
            
            # Reject token, update global_p and break
            else:
                global_p[cur_ind] = new_p
                break
        
        # Generate bonus token, don't generate if eos token is the last token
        if sampled_tokens.size(0) == 0 or sampled_tokens[-1] != self.ssm.eos_token_id:
            #TODO: Remove unnecessary shape modification operations
            if not do_sample:
                bonus_token = global_p[cur_ind].argmax()[None]
            else:
                bonus_token = global_p[cur_ind].multinomial(num_samples=1).squeeze(-1)

            sampled_tokens = torch.cat([sampled_tokens, bonus_token])
            hidden_indices = torch.cat([hidden_indices, cur_ind])
        
        return sampled_tokens[None], hidden_indices, (total_len, accept_len)

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
        llm_max_cache_len = stopping_criteria.max_length + self.draft_params.max_verify_tokens # Add extra space for verifying
        llm_past_key_values = self.create_kv_cache(
            max_cache_len=llm_max_cache_len,
            max_batch_size=1,
            config=self.llm.model.config,
            device=input_ids.device,
            dtype=self.llm.model.dtype,
        )
        ssm_max_cache_len = stopping_criteria.max_length + self.draft_params.max_sample_tokens # Add extra space for sampling
        ssm_past_key_values = self.create_kv_cache(
            max_cache_len=ssm_max_cache_len,
            max_batch_size=1,
            config=self.ssm.model.config,
            device=input_ids.device,
            dtype=self.ssm.model.dtype,
        )
        tree_mask_base = torch.zeros(
            [1, 1, self.draft_params.max_verify_tokens, llm_max_cache_len],
            device=input_ids.device,
            dtype=torch.bool,
        )
        cache_position = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        # * prefill stage
        with nvtx.annotate("prefill", color="orange"):
            outputs = self.llm(input_ids, past_key_values=llm_past_key_values, output_hidden_states=True, cache_position=cache_position)
            
            next_idx = cache_position[-1] + 1
            cache_position = torch.arange(next_idx, next_idx+self.draft_params.max_verify_tokens, dtype=torch.long, device=input_ids.device)
        
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        # We keep the seq_len axis considering cases of multiple tokens.
        next_token_logits = outputs.logits[:, -1:, :].clone() # hf uses outputs.logits[:, -1, :] instead
        hidden_states = outputs.hidden_states[-1].clone()

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs
        
        next_tokens = self._sample_token(next_token_logits, logits_warper, do_sample)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        with nvtx.annotate("decoding"):
            finished = False
            while not finished:
                torch.compiler.cudagraph_mark_step_begin()
                
                # * speculate
                with nvtx.annotate("speculate", color="cyan"):
                    tree = self._speculate(input_ids, hidden_states, ssm_past_key_values, max_cache_len=ssm_max_cache_len)

                # * tree decoding
                with nvtx.annotate("tree_decoding", color="orange"):
                    prev_kv_len = llm_past_key_values.get_seq_length()
                    outputs = self._tree_decoding(tree, tree_mask_base, llm_past_key_values, position_offset=input_ids.shape[1]-1, cache_position=cache_position, device=hidden_states.device)
                
                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # We keep the seq_len axis considering cases of multiple tokens.
                next_token_logits = outputs.logits.clone()
                hidden_states = outputs.hidden_states[-1].clone()
                
                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                del outputs

                # * verify
                with nvtx.annotate("verify"):
                    sampled_tokens, hidden_indices, _ = self._verify(
                                                        tree, next_token_logits, 
                                                        logits_warper,
                                                        do_sample
                                                    )
                    
                    sampled_tokens = sampled_tokens.to(input_ids.device, non_blocking=True)
                    hidden_indices = hidden_indices.to(hidden_states.device, non_blocking=True)
                

                # * update input_ids, hidden_states, kv-cache, and cache_position
                with nvtx.annotate("update_data"):
                    input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
                    hidden_states = hidden_states[:, hidden_indices].clone()
                    llm_past_key_values.reorder_cache_with_offset(hidden_indices, offset=prev_kv_len, new_chunk_len=self.draft_params.max_verify_tokens, dim=2)
                    cache_position += sampled_tokens.shape[1]
                
                # * check stopping criteria
                finished = stopping_criteria(input_ids, None)
                
        return input_ids
    

class ProfileSDWrapper(SDWrapper):
    def __init__(self, out_dir="specdecodes/experiments/profile_data", prefix="sd", *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        self.profile_data = {}
        self.sampled_count = 1 # assume first token is sampled (prefill stage)
        self.iter_count = 1 # assume first step is done (prefill stage)
        
        self.out_dir = out_dir
        self.prefix = prefix
        
        self.exp_log = {}
        self.draft_time_per_iter = []
        self.target_time_per_iter = []
        self.verify_time_per_iter = []
        
        self.disable_logging = False
        
    def _speculate(self, input_ids, hidden_states, past_key_values, max_cache_len=None):
        if self.disable_logging:
            return super()._speculate(input_ids, hidden_states, past_key_values, max_cache_len)
        
        start_time = time.perf_counter()
        root = super()._speculate(input_ids, hidden_states, past_key_values, max_cache_len)
        self.draft_time_per_iter.append(time.perf_counter()-start_time)
        return root
    
    def _tree_decoding(self, tree, tree_mask_base, past_key_values, position_offset, cache_position, device):
        if self.disable_logging:
            return super()._tree_decoding(tree, tree_mask_base, past_key_values, position_offset, cache_position, device)
        
        start_time = time.perf_counter()
        outputs = super()._tree_decoding(tree, tree_mask_base, past_key_values, position_offset, cache_position, device)
        self.target_time_per_iter.append(time.perf_counter()-start_time)
        return outputs
    
    def _verify(self, tree, logits, logits_warper, do_sample):
        if self.disable_logging:
            return super()._verify(tree, logits, logits_warper, do_sample)
        
        start_time = time.perf_counter()
        sampled_tokens, hidden_indices, (total_len, accept_len) = super()._verify(tree, logits, logits_warper, do_sample)
        self.verify_time_per_iter.append(time.perf_counter()-start_time)
        
        # tokenize id to text for visualization
        # nodes = list(preorder_iter(root))
        # for node in nodes:
        #     node.id = self.tokenizer.decode(torch.tensor([node.id]), clean_up_tokenization_spaces=False)
        
        # profile data
        # json_graph = tree_to_nested_dict(root, name_key="name", attr_dict={"id": "id", "prob": "prob", "global_prob": "global_prob"})
        # sampled_tokens_list = sampled_tokens.squeeze(0).tolist()
        # self.profile_data[self.iter_count] = {}
        # self.profile_data[self.iter_count]["draft_tree"] = json_graph
        # self.profile_data[self.iter_count]["sampled_tokens"] = sampled_tokens_list
        
        # create profile data if not exist
        self.profile_data['iter'] = self.profile_data.get('iter', [])
        self.profile_data['total_len'] = self.profile_data.get('total_len', [])
        self.profile_data['accept_len'] = self.profile_data.get('accept_len', [])
            
        sampled_tokens_list = sampled_tokens.squeeze(0).tolist()
        self.profile_data['iter'].append(sampled_tokens_list)
        self.profile_data['total_len'].append(total_len)
        self.profile_data['accept_len'].append(accept_len)
        # logging
        logging.debug(
            f"Total: {tree.size()},"\
            f"\tPredicted ({accept_len}/{total_len}): {self.tokenizer.batch_decode(sampled_tokens.squeeze(0), clean_up_tokenization_spaces=False)}"
        )
        
        # update stats
        self.sampled_count += len(sampled_tokens[0])
        self.iter_count += 1
        
        return sampled_tokens, hidden_indices, (total_len, accept_len)
    
    
    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_warper: LogitsWarper,
        do_sample: bool,
    ):
        if self.disable_logging:
            return super()._generate(input_ids, stopping_criteria, logits_warper, do_sample)
        
        cur_time = time.strftime("%Y%m%d-%H%M%S")
        # prepare output directory
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            out_path = os.path.join(self.out_dir, f"{self.prefix}_{cur_time}.json")
        else:
            out_path = None
        
        # run generation
        org_input_len = len(input_ids[0])
        start_time = time.perf_counter()
        input_ids = super()._generate(input_ids, stopping_criteria, logits_warper, do_sample)
        end_time = time.perf_counter()
        
        # compute stats
        total_sampled = self.sampled_count
        total_iterations = self.iter_count
        avg_sampled = total_sampled / total_iterations
        depth = max(self.profile_data['total_len']) + 1
        
        # alpha (node)
        total_lens = torch.bincount( torch.tensor(self.profile_data['total_len']), minlength=depth)
        accept_lens = torch.bincount( torch.tensor(self.profile_data['accept_len']), minlength=depth)
        depth_total_cnt = total_lens + total_lens.sum() - total_lens.cumsum(dim=-1) # reverse cumsum
        depth_total_cnt = depth_total_cnt[1:] # remove first element
        depth_accept_cnt = accept_lens + accept_lens.sum() - accept_lens.cumsum(dim=-1) # reverse cumsum
        depth_accept_cnt = depth_accept_cnt[1:] # remove first element
        alpha_per_node = depth_accept_cnt.float() / depth_total_cnt.float()
        
        # aLive ratio
        depth_alive_rate = depth_total_cnt.float() / depth_total_cnt[0]
        
        # alpha (depth)
        sampled_lens = torch.tensor([len(sampled_tokens) for sampled_tokens in self.profile_data["iter"]])
        sampled_len_bins = torch.bincount(sampled_lens, minlength=depth+1)
        depth_total_cnt = sampled_len_bins + sampled_len_bins.sum() - sampled_len_bins.cumsum(dim=-1) # reverse cumsum
        depth_accept_cnt = depth_total_cnt - sampled_len_bins
        depth_total_cnt = depth_total_cnt[1:depth]
        depth_accept_cnt = depth_accept_cnt[1:depth]
        alpha_per_depth = depth_accept_cnt.float() / depth_total_cnt.float()
        
        # log stats
        tb = pt.PrettyTable()
        tb.field_names = [ "Summary \ Depth" ] + [ f"{i}" for i in range(1, depth) ]
        tb.add_row([ "Trials count" ] + [ f"{val}" for val in depth_total_cnt.tolist() ])
        tb.add_row([ "Accept count" ] + [ f"{val}" for val in depth_accept_cnt.tolist() ])
        tb.add_row([ "Alpha (node)" ] + [ f"{val:.2f}" for val in alpha_per_node.tolist() ])
        tb.add_row([ "Alpha (depth)" ] + [ f"{val:.2f}" for val in alpha_per_depth.tolist() ])
        tb.add_row([ "Alive ratio" ] + [ f"{val:.2f}" for val in depth_alive_rate.tolist() ])
        logging.info(
            f"Total sampled: {total_sampled},"\
            f"\tTotal iterations: {total_iterations},"\
            f"\tAverage sampled: {avg_sampled:.2f}"\
            f"\n{tb}"
        )
        
        # save profile data
        self.profile_data["total_sampled"] = total_sampled
        self.profile_data["total_iterations"] = total_iterations
        self.profile_data["average_sampled"] = avg_sampled
        if self.out_dir is not None:
            with open(out_path, "w") as f:
                json.dump(self.profile_data, f)
                
        # save exp_log
        self.exp_log['avg_draft_time'] = np.mean(self.draft_time_per_iter)
        self.exp_log['avg_target_time'] = np.mean(self.target_time_per_iter)
        self.exp_log['avg_verify_time'] = np.mean(self.verify_time_per_iter)
        self.exp_log['avg_sampled'] = avg_sampled
        self.exp_log['n_tokens'] = len(input_ids[0][org_input_len:])
        self.exp_log['tput'] = len(input_ids[0][org_input_len:]) / (end_time-start_time)
        logging.info(
            f"Average draft time: {self.exp_log['avg_draft_time']:.4f},"\
            f"\tAverage target time: {self.exp_log['avg_target_time']:.4f},"\
            f"\tAverage verify time: {self.exp_log['avg_verify_time']:.4f}"
            f"\nGenerated {self.exp_log['n_tokens']} tokens in {end_time-start_time:.2f}s, throughput: {self.exp_log['tput']:.2f} tokens/s"
        )
        
        return input_ids