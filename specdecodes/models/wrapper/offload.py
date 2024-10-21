import json
import logging
import os
import time
import torch
import torch.nn.functional as F
from .base import WrapperBase
from .sd import SDWrapper
import numpy as np
import logging
import gc

from transformers import AutoModelForCausalLM
from transformers.generation.logits_process import LogitsWarper
from transformers.generation.stopping_criteria import StoppingCriteria
from accelerate import dispatch_model
from transformers.cache_utils import StaticCache, DynamicCache

from bigtree import preorder_iter, levelorder_iter
from bigtree import tree_to_nested_dict
from ..utils import TreeDynamicCache, build_tree_attention_data

class OffloadWrapper(WrapperBase):
    def __init__(self):
        super(OffloadWrapper, self).__init__()

    def set_offload_llm(self, llm_path, memory_limit=8.0, device="cuda:0"):
        device_map = {
            "model.embed_tokens": "cuda:0",
            "model.rotary_emb": "cuda:0",
            "model.norm": "cuda:0",
            "lm_head": "cuda:0",
        }     

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, 
            device_map="cpu", 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.float16
        )

        estimated_mem = 0.0
        for param in self.llm.model.embed_tokens.parameters():
            estimated_mem += param.numel() * param.element_size()
        for param in self.llm.lm_head.parameters():
            estimated_mem += param.numel() * param.element_size()
        estimated_mem = estimated_mem / (1024 ** 3)
        
        decoder_layer_mem = 0.0
        for param in self.llm.model.layers[0].parameters():
            decoder_layer_mem += param.numel() * param.element_size()

        decoder_layer_mem = decoder_layer_mem / (1024 ** 3)

        # TODO: Check the memory usage to check how much layers to be offloaded
        for i in range(len(self.llm.model.layers)):
            if estimated_mem <= memory_limit - 2 * decoder_layer_mem - 0.5:
                estimated_mem += decoder_layer_mem
                device_map[f"model.layers.{i}"] = device
            else:
                device_map[f"model.layers.{i}"] = "cpu"
        
        # set pin_memory to reduce memory access time
        for layer in self.llm.model.layers:
            for param in layer.parameters():
                param.data = param.data.cpu().pin_memory(device)

        self.llm = dispatch_model(self.llm, device_map=device_map)
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
        logging.debug(f"Allocated Memory = {allocated_memory} GB")
        
        if allocated_memory > memory_limit:
            logging.info(f"[Warning] memory usage is too much")

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
        org_input_len = len(input_ids[0])

        # * prepare kv-cache
        llm_past_key_values = DynamicCache()
        
        # * prefill stage
        outputs = self.llm(input_ids, past_key_values=llm_past_key_values, return_dict=True)
        
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        # We keep the seq_len axis considering cases of multiple tokens.
        next_token_logits = outputs.logits[:, -1:, :].clone() # hf uses outputs.logits[:, -1, :].clone() here

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs
        
        next_tokens = self._sample_token(next_token_logits, logits_warper, do_sample)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        finished = False
        start_time = time.perf_counter()
        while not finished:
            outputs = self.llm(input_ids[:, -1:], past_key_values=llm_past_key_values, return_dict=True)
        
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # We keep the seq_len axis considering cases of multiple tokens.
            next_token_logits = outputs.logits.clone()
            
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
            
            next_tokens = self._sample_token(next_token_logits, logits_warper, do_sample)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # Stopping criteria
            finished = stopping_criteria(input_ids, None)
        
        end_time = time.perf_counter()
        n_gen_token = len(input_ids[0][org_input_len:])
        logging.info(f"Inference Speed of {self.llm.model.config._name_or_path}: {n_gen_token / (end_time-start_time)} token/s")

        return input_ids

class OffloadSDWrapper(SDWrapper):
    def __init__(self, method="greedy"):
        super(OffloadSDWrapper, self).__init__(method=method)

    def set_offload_llm(self, llm_path, memory_limit=8.0, device="cuda:0"):
        assert self.ssm is not None, "SSM model must first be loaded on gpu"
        device_map = {
            "model.embed_tokens": "cuda:0",
            "model.rotary_emb": "cuda:0",
            "model.norm": "cuda:0",
            "lm_head": "cuda:0",
        }

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, 
            device_map="cpu", 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.float16
        )

        estimated_mem = torch.cuda.memory_allocated(device)
        for param in self.llm.model.embed_tokens.parameters():
            estimated_mem += param.numel() * param.element_size()
        for param in self.llm.lm_head.parameters():
            estimated_mem += param.numel() * param.element_size()
        estimated_mem = estimated_mem / (1024 ** 3)
        
        decoder_layer_mem = 0.0
        for param in self.llm.model.layers[0].parameters():
            decoder_layer_mem += param.numel() * param.element_size()

        decoder_layer_mem = decoder_layer_mem / (1024 ** 3)

        # TODO: Check the memory usage to check how much layers to be offloaded
        for i in range(len(self.llm.model.layers)):
            if estimated_mem <= memory_limit - 2 * decoder_layer_mem - 0.5:
                estimated_mem += decoder_layer_mem
                device_map[f"model.layers.{i}"] = device
            else:
                device_map[f"model.layers.{i}"] = "cpu"
        
        # set pin_memory to reduce memory access time
        for layer in self.llm.model.layers:
            for param in layer.parameters():
                param.data = param.data.cpu().pin_memory(device)

        self.llm = dispatch_model(self.llm, device_map=device_map)
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
        logging.debug(f"Allocated Memory = {allocated_memory} GB")
        
        if allocated_memory > memory_limit:
            logging.info(f"[Warning] memory usage is too much")

        
    def _speculate(self, inputs, past_key_values):
        return self.ssm.speculate(
            inputs,
            past_key_values=past_key_values
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
            return_dict=True
        )

        return outputs

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
        org_input_len = len(input_ids[0])

        # * prepare kv-cache
        llm_past_key_values = TreeDynamicCache()
        ssm_past_key_values = TreeDynamicCache()

        # * prefill stage
        outputs = self.llm(input_ids, past_key_values=llm_past_key_values, return_dict=True)
        
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1:, :].clone() # hf uses outputs.logits[:, -1, :] instead

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

        next_tokens = self._sample_token(next_token_logits, logits_warper, do_sample)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        
        speculated_tokens_per_iter = []
        draft_time_per_iter = []
        target_time_per_iter = []

        finished = False
        start_time = time.perf_counter()
        while not finished:
            # * speculate
            draft_start = time.perf_counter()
            root = self._speculate(input_ids, ssm_past_key_values)
            draft_time_per_iter.append(time.perf_counter()-draft_start)

            # * tree decoding
            prev_kv_len = llm_past_key_values.get_seq_length()

            target_start = time.perf_counter()
            outputs = self._tree_decoding(root, llm_past_key_values, position_offset=input_ids.shape[1]-1, device=input_ids.device, dtype=self.llm.dtype)
            
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits

            # * verify
            sampled_tokens, hidden_indices, _ = self._verify(
                                                root, next_token_logits, 
                                                logits_warper,
                                                do_sample
                                            )
            target_time_per_iter.append(time.perf_counter()-target_start)

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs, root
            gc.collect()
            
            sampled_tokens = sampled_tokens.to(input_ids.device)
            hidden_indices = hidden_indices.to(input_ids.device)

            speculated_tokens_per_iter.append(len(sampled_tokens[0]))
            logging.info(f"new speculated number of token: +{len(sampled_tokens[0])}")

            # * update input_ids, hidden_states, and kv-cache
            # llm_past_key_values.crop(prev_kv_len)
            input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
            llm_past_key_values.reorder_cache_with_offset(hidden_indices, offset=prev_kv_len, dim=2)

            # * check stopping criteria
            finished = stopping_criteria(input_ids, None)
        
        end_time = time.perf_counter()

        mean_gen_rate = np.mean(speculated_tokens_per_iter)
        mean_draft_time = np.mean(draft_time_per_iter)
        mean_target_time = np.mean(target_time_per_iter)
        logging.info(f"Average spculated number of token: {mean_gen_rate} tokens")
        logging.info(f"Average draft model time: {mean_draft_time}s")
        logging.info(f"Average target model time: {mean_target_time}s")
        logging.info(f"Theoretically speedup: {mean_gen_rate * mean_target_time / (mean_target_time + mean_draft_time)}")

        n_gen_tokens = len(input_ids[0][org_input_len:])
        logging.info(f"Generate speed: {n_gen_tokens/(end_time-start_time)} tok/s")

        return input_ids    

class ProfileOffloadSDWrapper(OffloadSDWrapper):
    def __init__(self, method="greedy"):
        super(ProfileOffloadSDWrapper, self).__init__(method=method)

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
        org_input_len = len(input_ids[0])

        # * prepare kv-cache
        llm_past_key_values = TreeDynamicCache()
        ssm_past_key_values = TreeDynamicCache()

        # * prefill stage
        outputs = self.llm(input_ids, past_key_values=llm_past_key_values, return_dict=True)
        
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1:, :].clone() # hf uses outputs.logits[:, -1, :] instead

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

        next_tokens = self._sample_token(next_token_logits, logits_warper, do_sample)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        
        speculated_tokens_per_iter = []
        draft_time_per_iter = []
        target_time_per_iter = []

        finished = False
        start_time = time.perf_counter()
        while not finished:
            # * speculate
            draft_start = time.perf_counter()
            root = self._speculate(input_ids, ssm_past_key_values)
            draft_time_per_iter.append(time.perf_counter()-draft_start)

            # * tree decoding
            prev_kv_len = llm_past_key_values.get_seq_length()

            target_start = time.perf_counter()
            outputs = self._tree_decoding(root, llm_past_key_values, position_offset=input_ids.shape[1]-1, device=input_ids.device, dtype=self.llm.dtype)
            
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits

            # * verify
            sampled_tokens, hidden_indices, _ = self._verify(
                                                root, next_token_logits, 
                                                logits_warper,
                                                do_sample
                                            )
            target_time_per_iter.append(time.perf_counter()-target_start)

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs, root
            gc.collect()
            
            sampled_tokens = sampled_tokens.to(input_ids.device)
            hidden_indices = hidden_indices.to(input_ids.device)

            speculated_tokens_per_iter.append(len(sampled_tokens[0]))
            logging.debug(f"new speculated number of token: +{len(sampled_tokens[0])}")

            # * update input_ids, hidden_states, and kv-cache
            # llm_past_key_values.crop(prev_kv_len)
            input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
            llm_past_key_values.reorder_cache_with_offset(hidden_indices, offset=prev_kv_len, dim=2)

            # * check stopping criteria
            finished = stopping_criteria(input_ids, None)
        
        end_time = time.perf_counter()

        mean_gen_rate = np.mean(speculated_tokens_per_iter)
        mean_draft_time = np.mean(draft_time_per_iter)
        mean_target_time = np.mean(target_time_per_iter)
        n_gen_tokens = len(input_ids[0][org_input_len:])

        log = {}
        log["accept_rate"] = mean_gen_rate
        log["draft_time"] = mean_draft_time
        log["target_time"] = mean_target_time
        log["n_gen_tokens"] = n_gen_tokens
        log["tput"] = n_gen_tokens / (end_time-start_time)

        return input_ids, log            