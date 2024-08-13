import logging
import torch
from .base import WrapperBase

from transformers.generation.logits_process import LogitsWarper
from transformers.generation.stopping_criteria import StoppingCriteria

from bigtree import preorder_iter, levelorder_iter, shift_nodes, find_attrs
from ..utils import TreeDynamicCache, build_tree_attention_data 

class SDWrapper(WrapperBase):
    def __init__(self):
        super(SDWrapper, self).__init__()
  
    def set_ssm(self, ssm):
        self.ssm = ssm
    
    def _speculate(self, hidden_states, input_ids, past_key_values, eos_token_id=None):
        return self.ssm.speculate(
            hidden_states,
            input_ids,
            embed_tokens=self.llm.get_input_embeddings(), 
            lm_head=self.llm.lm_head,
            past_key_values=past_key_values,
            eos_token_id=eos_token_id,
        )
        
    def _tree_decoding(self, root, past_key_values, position_offset, device, dtype=torch.float32):
        # Preparing llm's tree decoding data
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
        accept_tokens = []
        hidden_indices = []
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
                        accept_tokens.append(child.id) # real_token_ids[cur.ind]
                        hidden_indices.append(cur.ind)
                        cur = child
                        break
                    
                # stop loop if no token is accepted
                if accepts_token == False: break
            
            if len(accept_tokens) == 0 or accept_tokens[-1] != eos_token_id: # no token matched, accept first token
                bonus_token = real_token_ids[cur.ind].item()
                accept_tokens.append(bonus_token)
                hidden_indices.append(cur.ind)

        elif sampling_method == "eagle":
            assert do_sample == True, "Eagle method requires sampling"
            global_gtp = self._sample_token(logits, logits_warper, do_sample=True, return_probs=True).squeeze(0) # remove batch dim
            
            cur = root
            while cur.children:
                gtp = global_gtp[cur.ind]
                
                accepts_token = False
                for child in cur.children:
                    r = torch.rand(1).item()
                    # px = gtp[child.id]
                    # qx = 1  # in this iteration, only child.id' prob is 1, others are 0.
                    # if r <= px / qx:
                    if r <= gtp[child.id]: # since qx = 1, we can compare r with px directly
                        accepts_token = True
                        accept_tokens.append(child.id)
                        hidden_indices.append(cur.ind)
                        cur = child
                        break
                    else:
                        # gpt = max(gpt - childprob, 0)
                        gtp[child.id] = 0 # only child.id' prob is 1, equivalent to function above
                        gtp = gtp / gtp.sum()
                        # childprob[child.id] = 0
                        # childprob = childprob / childprob.sum()
                        
                # stop loop if no token is accepted
                if accepts_token == False: break
                
            if len(accept_tokens) == 0 or accept_tokens[-1] != eos_token_id: # eos token should be the last token
                bonus_token = torch.multinomial(global_gtp[cur.ind], 1).item()
                accept_tokens.append(bonus_token)
                hidden_indices.append(cur.ind)
        
        elif sampling_method == "sequoia":
            raise NotImplementedError("Sequoia method is not implemented yet")
        
        else:
            raise ValueError("Invalid method")
        
        return torch.tensor([accept_tokens]), torch.tensor(hidden_indices) #! add back batch size dim to accept_tokens

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
            1. Perform speculative sampling with the SSM.
            2. Conduct tree decoding with the language model (LLM).
            3. Verify the candidates by accepting or rejecting them.
            4. Update the key-value cache, input_ids, and hidden_states accordingly.

        Args:
            input_ids (torch.LongTensor): The input token IDs.
            stopping_criteria (StoppingCriteria): The criteria to stop the generation.
            logits_warper (LogitsWarper): The warper to modify the logits.
            do_sample (bool): Whether to sample tokens during generation.

        Returns:
            input_ids (torch.LongTensor): The generated token IDs.
        """
        assert self.llm is not None, "LLM model must be provided"
        assert self.ssm is not None, "SSM model must be provided"
        assert self.tokenizer is not None, "Tokenizer must be provided"

        # * clone input_ids 
        input_ids = input_ids.clone()
        org_input_len = input_ids.shape[1]

        # * prepare kv-cache
        llm_past_key_values = TreeDynamicCache()
        ssm_past_key_values = TreeDynamicCache()

        # * prefill stage
        outputs = self.llm(input_ids, past_key_values=llm_past_key_values, output_hidden_states=True)
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1:].clone() #TODO: check shape, hf uses outputs.logits[:, -1, :].clone()
        hidden_states = outputs.hidden_states[-1].clone()

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs
        
        next_tokens = self._sample_token(next_token_logits, logits_warper, do_sample)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        finished = False
        total_accepted = 0
        total_steps = 0
        while not finished:
            # * speculate
            root = self._speculate(hidden_states, input_ids, ssm_past_key_values, eos_token_id=self.tokenizer.eos_token_id)

            # * tree decoding
            prev_kv_len = llm_past_key_values.get_seq_length()
            outputs = self._tree_decoding(root, llm_past_key_values, position_offset=input_ids.shape[1]-1, device=input_ids.device, dtype=hidden_states.dtype)
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits
            hidden_states = outputs.hidden_states[-1].clone()
            
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

            # * verify
            accept_tokens, hidden_indices = self._verify(
                                                root, next_token_logits, 
                                                logits_warper, 
                                                do_sample,
                                                eos_token_id=self.tokenizer.eos_token_id,
                                                sampling_method="eagle" if do_sample else "naive",
                                                # sampling_method="naive",
                                            )
            logging.debug(
                f"Total: {len(list(preorder_iter(root)))},"\
                f"\tPredicted: {self.tokenizer.batch_decode(accept_tokens.squeeze(0), clean_up_tokenization_spaces=False)}"
            )
            total_accepted += len(accept_tokens[0])
            total_steps += 1
            
            accept_tokens = accept_tokens.to(input_ids.device)
            hidden_indices = hidden_indices.to(hidden_states.device)

            # * update input_ids, hidden_states, and kv-cache
            # llm_past_key_values.crop(input_ids.shape[1])
            input_ids = torch.cat([input_ids, accept_tokens], dim=-1)
            hidden_states = hidden_states[:, hidden_indices].clone()
            llm_past_key_values.reorder_cache_with_offset(hidden_indices, offset=prev_kv_len, dim=2)
            # ssm_past_key_values.reorder_cache_with_offset(hidden_indices, offset=prev_kv_len, dim=2)
            
            # * check stopping criteria
            finished = stopping_criteria(input_ids, None)
            
            # print(f'used time: {(time.time() - st) / num_repeats * 1000} ms')
            # used_mem = torch.cuda.max_memory_allocated()
            # print(f'peak mem: {used_mem / 1024 ** 3} GB')
        
        logging.info(f"Total accepted: {total_accepted}, Total iterations: {total_steps}, Average accepted: {total_accepted/total_steps:.2f}")
        return input_ids