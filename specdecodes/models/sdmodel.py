import torch
import torch.nn as nn


from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation.logits_process import LogitsWarper, StoppingCriteria
from transformers.generation.logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, LogitNormalization
from transformers.generation.logits_process import StoppingCriteriaList, MaxLengthCriteria, MaxTimeCriteria, EosTokenCriteria

from .utils import build_tree_attention_data


class TreeDynamicCache(DynamicCache):
    def reorder_cache(self, beam_idx: torch.LongTensor, dim=0):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(dim, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(dim, beam_idx.to(device))

    def reorder_cache_with_offset(self, beam_idx: torch.LongTensor, offset=0, dim=0):
        """Reorders the cache for beam search, given the selected beam indices, while [:offset] remain unchanged""" 
        beam_idx = torch.cat([torch.arange(offset), beam_idx + offset], dim=0)
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(dim, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(dim, beam_idx.to(device))

# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
# Simplified several functions from class GenerationMixin
class SimpleWrapper(nn.Module):
    def __init__(self):
        super(SimpleWrapper, self).__init__()
    
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
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k))
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p))
        
        warpers = LogitsProcessorList()
        return None
    
    def _get_stopping_criteria(
        self,
        max_length: int = None,
        max_time: float = None,
        eos_token_tensor: torch.LongTensor = None,
    ):
        criteria = StoppingCriteriaList()
        if max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=max_time))
        if eos_token_tensor is not None:
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
    ):
        r"""
        This method is expected to be implemented by subclasses.
        """
        raise NotImplementedError

    def set_llm(self, llm):
        self.llm = llm    
        
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
        stopping_criteria = self._get_stopping_criteria(max_length=max_length)
        
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

class SpecDecodesWrapper(SimpleWrapper):
    def __init__(self):
        super(SpecDecodesWrapper, self).__init__()
  
    def set_ssm(self, ssm):
        self.ssm = ssm
    
    def _speculate(self, input_ids, hidden_states, past_key_values):
        self.ssm.speculate(input_ids, hidden_states, past_key_values)
        
    def _tree_decoding(self, root, outputs, past_key_values, position_offset, device):
        # Preparing llm's tree decoding data
        tree_input_ids, tree_position_ids, tree_mask = build_tree_attention_data(root, position_offset=position_offset)
        
        # Move to device
        tree_input_ids = tree_input_ids.to(device)
        tree_position_ids = tree_position_ids.to(device)
        tree_mask = tree_mask.to(device)
        
        # llm forward
        outputs = self.llm(
            tree_input_ids,
            output_orig=True,
            past_key_values=past_key_values,
            attention_mask=tree_mask,
            position_ids=tree_position_ids,
        )
        return outputs
    
    def _verify(self, root, logits, logits_warper, method="greedy"):
        #TODO: process logits
        accept_indices = [root.ind] # tree.ind (first token, generated by llm), is already included in input_ids, so no need to accept it again
        hidden_indices = []
        if method == "greedy":
            predicted_tokens = self._sample_token(logits, logits_warper, do_sample=False)
            # for each depth, find token that matches the predicted token
            cur = root
            while cur.children:
                p_token = predicted_tokens[cur.ind]
                for child in cur.children:
                    if child.id == p_token:
                        accept_indices.append(child.ind)
                        hidden_indices.append(cur.ind)
                        cur = child
                        break
                else: # iterated all childrens, but none is accepted
                    break

            # bonus_token = predicted_tokens[None, cur.ind] # bonus token, sampled from prob of last accepted token
            accept_tokens = predicted_tokens[accept_indices]
            hidden_indices.append(cur.ind)

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

        # else:
        #     raise ValueError("Invalid method")

        return accept_tokens[None], torch.tensor(hidden_indices) #! [None] to add back batch size dim

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_warper: LogitsWarper,
        do_sample: bool,
    ):
        #* prefill stage
        # llm forward
        # sample token, append to input_ids

        #* decode stage (with speculative decoding)
        # for loop
        #   1. ssm speculate
        #   2. llm tree decoding
        #   3. verify (accept or reject candidates)
        #   4. update kv-cache, input_ids, and hidden_states


        assert self.llm is not None, "LLM model must be provided"
        assert self.ssm is not None, "SSM model must be provided"
        assert stopping_criteria is not None, "Stopping criteria must be provided"

        # * clone input_ids 
        input_ids = input_ids.clone()

        # * prepare kv-cache
        llm_past_key_values = TreeDynamicCache()
        ssm_past_key_values = TreeDynamicCache()

        # * prefill stage
        outputs = self.llm(input_ids, past_key_values=llm_past_key_values, output_hidden_states=True)
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].clone()
        hidden_states = outputs.hidden_states.clone()
        
        next_tokens = self._sample_token(next_token_logits, logits_warper, do_sample)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        finished = False
        while not finished:
            # * speculate
            root = self._speculate(input_ids, hidden_states, ssm_past_key_values)

            # * tree decoding
            prev_kv_len = llm_past_key_values.get_seq_length()
            outputs = self._tree_decoding(root, outputs, llm_past_key_values, position_offset=input_ids.shape[1]-1)
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()
            hidden_states = outputs.hidden_states[-1].clone()

            # * verify
            accept_tokens, hidden_indices = self._verify(root, outputs)

            # * update kv-cache, input_ids, and hidden_states
            llm_past_key_values.reorder_cache_with_offset(hidden_indices, offset=prev_kv_len, dim=2)
            input_ids = torch.cat([input_ids, accept_tokens], dim=-1)
            hidden_states = hidden_states[:, hidden_indices]

            # * check stopping criteria
            finished = stopping_criteria(input_ids, None)


