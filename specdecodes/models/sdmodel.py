import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsWarper, LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, LogitNormalization
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, MaxLengthCriteria, MaxTimeCriteria, EosTokenCriteria

from transformers.cache_utils import StaticCache, DynamicCache

from .utils import TreeDynamicCache, build_tree_attention_data 

# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
# Several functions are form class GenerationMixin, simplified.
class SimpleWrapper(nn.Module):
    def __init__(self):
        super(SimpleWrapper, self).__init__()
    
    def set_llm(self, llm):
        self.llm = llm
        
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        
    def _get_logits_warper(
        self,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ):
        """
        Simplified HuggingFace's `LogitsProcessorList` for multinomial sampling.
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """
        # instantiate warpers list
        warpers = LogitsProcessorList()
        
        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k))
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p))
        
        return warpers
    
    def _get_stopping_criteria(
        self,
        max_length: int = None,
        max_time: float = None,
        eos_token_tensor: torch.LongTensor = None,
    ):
        criteria = StoppingCriteriaList()
        if max_length is not None:
            max_position_embeddings = getattr(self.llm.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=max_time))
        if eos_token_tensor is not None:
            # EosTokenCriteria only checks last input token,
            # make sure not token is appended after eos_token_tensor during generation
            criteria.append(EosTokenCriteria(eos_token_id=eos_token_tensor))
        return criteria
    
    def _sample_token(
        self,
        logits: torch.FloatTensor,
        logits_warper: LogitsWarper,
        do_sample: bool,
        return_probs: bool = False,
    ):
        if do_sample:
            batch, seq_len, vocab_size = logits.shape
            
            logits = logits.view(-1, vocab_size)
            next_token_scores = logits_warper(None, logits)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            
            if return_probs:
                return probs.view(batch, seq_len, vocab_size) # preserve shape
            else:
                token = torch.multinomial(probs, 1)
                return token.view(batch, seq_len) # preserve shape
        else:
            if return_probs:
                return torch.softmax(logits, dim=-1)
            else:
                return torch.argmax(logits, dim=-1)

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_warper: LogitsWarper,
        do_sample: bool,
        *args,
        **kwargs,
    ):
        r"""
        This method is expected to be implemented by subclasses.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        temperature=None,
        top_p=None,
        top_k=None,
        max_length=2048,
        do_sample=True,
    ):        
        # 1. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(max_length=max_length, eos_token_tensor=self.tokenizer.eos_token_id)
        
        # 2. prepare logits warper (if `do_sample` is `True`)
        logits_warper = (
            self._get_logits_warper(
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k,
            ) if do_sample else None
        )
        
        # 3. generate
        results = self._generate(
            input_ids=input_ids,
            stopping_criteria=stopping_criteria,
            logits_warper=logits_warper,
            do_sample=do_sample,
        )
        return results

class HuggingFaceWrapper(SimpleWrapper):
    def __init__(self):
        super(HuggingFaceWrapper, self).__init__()
    
    def generate(
        self, 
        input_ids: torch.LongTensor, 
        temperature=None, top_p=None, top_k=None, 
        max_length=2048, do_sample=True, 
        *args, 
        **kwargs
    ):
        assert self.llm is not None, "LLM model must be provided"
        
        return self.llm.generate(
            input_ids=input_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_length=max_length,
            do_sample=do_sample,
            *args,
            **kwargs,
        )
        
class NaiveWrapper(SimpleWrapper):
    def __init__(self):
        super(NaiveWrapper, self).__init__()
 
    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_warper: LogitsWarper,
        do_sample: bool,
    ):
        assert self.llm is not None, "LLM model must be provided"

        # * clone input_ids 
        input_ids = input_ids.clone()

        # * prepare kv-cache
        llm_past_key_values = DynamicCache()
        
        # * prefill stage
        outputs = self.llm(input_ids, past_key_values=llm_past_key_values, return_dict=True)
        
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1:].clone() #TODO: check shape, hf uses outputs.logits[:, -1, :].clone()

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs
        
        next_tokens = self._sample_token(next_token_logits, logits_warper, do_sample)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        finished = False
        while not finished:
            outputs = self.llm(input_ids[:, -1:], past_key_values=llm_past_key_values, return_dict=True)
        
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()
            
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
            
            next_tokens = self._sample_token(next_token_logits, logits_warper, do_sample)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # * check stopping criteria
            finished = stopping_criteria(input_ids, None)
            
        return input_ids


class SDWrapper(SimpleWrapper):
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
            
            if cur.id != eos_token_id: # no token matched, accept first token
                bonus_token = real_token_ids[cur.ind]
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
                
            if cur.id != eos_token_id: # eos token should be the last token
                bonus_token = torch.multinomial(global_gtp[cur.ind], 1)
                accept_tokens.append(bonus_token)
                hidden_indices.append(cur.ind)
        
        elif sampling_method == "sequoia":
            raise NotImplementedError("Sequoia method is not implemented yet")
        
        else:
            raise ValueError("Invalid method")

        return torch.tensor(accept_tokens)[None], torch.tensor(hidden_indices) #! [None] to add back batch size dim

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
                                                sampling_method="eagle",
                                                # sampling_method="naive",
                                            )
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
        
        return input_ids


