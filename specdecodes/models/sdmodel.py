import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsWarper, LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, LogitNormalization
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, MaxLengthCriteria, MaxTimeCriteria, EosTokenCriteria

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
        
        return None
    
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
    ):
        if do_sample:
            next_token_scores = logits_warper(None, logits)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(1)
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
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
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

class NaiveWrapper(SimpleWrapper):
    def __init__(self):
        super(NaiveWrapper, self).__init__()
    
    def generate(
        self, 
        input_ids: torch.LongTensor, 
        temperature=0, top_p=0, top_k=0, 
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
    def _verify(self, root, logits, logits_warper, method="greedy", eos_token_id=None):
        accept_tokens = [] # (first token, generated by llm), is already included in input_ids, so no need to accept it again
        hidden_indices = []
        if method == "greedy":
            real_token_ids = self._sample_token(logits, logits_warper, do_sample=False)[0]

            # for each depth, find token that matches the predicted token
            # if all tokens are matched, then the last token is the bonus token
            cur = root
            while cur.children:
                llm_token_id = real_token_ids[cur.ind]
                for child in cur.children:
                    if child.id == llm_token_id:
                        accept_tokens.append(child.id) # real_token_ids[cur.ind]
                        hidden_indices.append(cur.ind)
                        cur = child
                        break
                else: # iterated all childrens, but none is accepted
                    break
            else: # all tokens matched, add bonus token
                if cur.id != eos_token_id: # eos token should be the last token
                    accept_tokens.append(real_token_ids[cur.ind])
                    hidden_indices.append(cur.ind)
                
            if len(accept_tokens) == 0: # no token matched, accept first token
                accept_tokens.append(real_token_ids[0])
                hidden_indices.append(0)

        elif method == "stochastic":
            raise NotImplementedError
        #     llm_probs = sampling_logit(logits, logits_processor, return_prob=True)
        #     cur = tree
        #     while cur.children:
        #         for child in cur.children:
        #             r = random.random()
        #             px = llm_probs[child.ind]
        #             qx = child.prob
        #             if r <= px / qx:
        #                 accept_indices.append(child.ind)
        #                 cur = child
        #                 break
        #             else:
        #                 #TODO: adjust llm probs
        #                 pass
                
        #         else: # iterated all childrens, but none is accepted
        #             break

        #     # sample the next token
        #     bonus_token = torch.multinomial(llm_probs, 1)[None]

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
            accept_tokens, hidden_indices = self._verify(root, next_token_logits, logits_warper, method="greedy", eos_token_id=self.tokenizer.eos_token_id)
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


