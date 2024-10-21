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

import prettytable as pt

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
    def __init__(self, method="greedy", out_dir="specdecodes/experiments/profile_data", prefix="sd"):
        super(ProfileOffloadSDWrapper, self).__init__(method=method)
        self.profile_data = {}
        self.sampled_count = 1 # assume first token is sampled (prefill stage)
        self.iter_count = 1 # assume first step is done (prefill stage)
        
        self.out_dir = out_dir
        self.prefix = prefix
        
    
    def _verify(self, root, logits, logits_warper, do_sample):
        sampled_tokens, hidden_indices, (total_len, accept_len) = super()._verify(root, logits, logits_warper, do_sample)
        
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
        
        if self.profile_data.get('total_len') is None:
            self.profile_data['total_len'] = []
        
        if self.profile_data.get('accept_len') is None:
            self.profile_data['accept_len'] = []
            
        sampled_tokens_list = sampled_tokens.squeeze(0).tolist()
        self.profile_data['iter'].append(sampled_tokens_list)
        self.profile_data['total_len'].append(total_len)
        self.profile_data['accept_len'].append(accept_len)
        # logging
        logging.debug(
            f"Total: {len(list(preorder_iter(root)))},"\
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
        cur_time = time.strftime("%Y%m%d-%H%M%S")
        
        # prepare output directory
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            out_path = os.path.join(self.out_dir, f"{self.prefix}_{cur_time}.json")
        else:
            out_path = None
        
        # run generation
        input_ids = super()._generate(input_ids, stopping_criteria, logits_warper, do_sample)
        
        
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
        
        return input_ids            