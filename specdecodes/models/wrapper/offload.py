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

from torch.profiler import profile, record_function, ProfilerActivity

from transformers import AutoModelForCausalLM
from transformers.generation.logits_process import LogitsWarper
from transformers.generation.stopping_criteria import StoppingCriteria
from accelerate import dispatch_model
# from .my.big_modeling import dispatch_model
from transformers.cache_utils import StaticCache, DynamicCache

from bigtree import preorder_iter, levelorder_iter
from bigtree import tree_to_nested_dict
from ..utils import TreeDynamicCache, build_tree_attention_data

from awq import AutoAWQForCausalLM
import prettytable as pt

class OffloadWrapper(WrapperBase):
    def __init__(self, pin_memory=True):
        super(OffloadWrapper, self).__init__()
        self.pin_memory = pin_memory

    def set_offload_llm(self, llm_path, memory_limit=4.0, device="cuda:0"):
        device_map = {
            "model.embed_tokens": "cuda:0",
            "model.rotary_emb": "cuda:0",
            "model.norm": "cuda:0",
            "lm_head": "cuda:0",
        }     
        logging.info(f'[Memory Limit]: {memory_limit} GB')
        
        if 'autoawq' in llm_path:
            memory_map = {0: "0GiB", "cpu": "99GiB"}
            self.llm = AutoAWQForCausalLM.from_quantized(
                llm_path, 
                fuse_layers=False,
                low_cpu_mem_usage=True,
                max_memory=memory_map
            ).model
            # self.llm = AutoModelForCausalLM.from_pretrained(
            #     llm_path, 
            #     device_map="cpu", 
            #     low_cpu_mem_usage=True
            # )
            import torch.nn as nn
            buffer_keywords = ["qweight", "qzeros", "scales"]
            for name, buffer in list(self.llm.named_buffers()):  # Use list() to avoid modification issues during iteration
                if any(keyword in name for keyword in buffer_keywords):
                    # Extract the parent module and attribute name
                    module_name, buffer_name = name.rsplit('.', 1)
                    parent_module = dict(self.llm.named_modules())[module_name]
                    
                    # Unregister the buffer
                    buffer_data = getattr(parent_module, buffer_name)
                    delattr(parent_module, buffer_name)  # Remove it from the module

                    # Register it as a trainable parameter
                    parent_module.register_parameter(buffer_name, nn.Parameter(buffer_data, requires_grad=False))

        else: 
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path, 
                device_map="cpu", 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.float16
            )

        estimated_mem = torch.cuda.memory_allocated(device)
        logging.info(f"Init Allocated Memory = {estimated_mem / (1024 ** 3)} GB")
        
        for param in self.llm.model.embed_tokens.parameters():
            estimated_mem += param.numel() * param.element_size()
        for buffer in self.llm.model.embed_tokens.buffers():
            estimated_mem += buffer.numel() * buffer.element_size()

        for param in self.llm.lm_head.parameters():
            estimated_mem += param.numel() * param.element_size()
        for buffer in self.llm.lm_head.buffers():
            estimated_mem += buffer.numel() * buffer.element_size()
        estimated_mem = estimated_mem / (1024 ** 3)
        
        decoder_layer_mem = 0.0
        for param in self.llm.model.layers[0].parameters():
            decoder_layer_mem += param.numel() * param.element_size()
        for buffer in self.llm.lm_head.buffers():
            decoder_layer_mem += buffer.numel() * buffer.element_size()

        decoder_layer_mem = decoder_layer_mem / (1024 ** 3)
        memory_limit = memory_limit / 1.2

        # TODO: Check the memory usage to check how much layers to be offloaded
        for i in range(len(self.llm.model.layers)):
            if estimated_mem <= memory_limit - decoder_layer_mem:
                estimated_mem += decoder_layer_mem
                device_map[f"model.layers.{i}"] = device
            else:
                device_map[f"model.layers.{i}"] = "cpu"
        logging.info(f"[Check] device_map: {device_map}")
        
        # set pin_memory to reduce memory access time
        if self.pin_memory:
            for layer in self.llm.model.layers:
                for param in layer.parameters():
                    param.data = param.data.cpu().pin_memory(device)

        self.llm = dispatch_model(self.llm, device_map=device_map)
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
        logging.info(f"Allocated Memory = {allocated_memory} GB")
        
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
        self.exp_log['tput'] = n_gen_token / (end_time-start_time)

        return input_ids

class OffloadSDWrapper(SDWrapper):
    def __init__(self, method="greedy"):
        super(OffloadSDWrapper, self).__init__(method=method)

    def set_offload_llm(self, llm_path, memory_limit=6.0, device="cuda:0"):
        assert self.ssm is not None, "SSM model must first be loaded on gpu"
        device_map = {
            "model.embed_tokens": "cuda:0",
            "model.rotary_emb": "cuda:0",
            "model.norm": "cuda:0",
            "lm_head": "cuda:0",
        }
        logging.info(f'[Memory Limit]: {memory_limit} GB')
        if 'autoawq' in llm_path:
            memory_map = {0: "0GiB", "cpu": "99GiB"}
            self.llm = AutoAWQForCausalLM.from_quantized(
                llm_path, 
                fuse_layers=False,
                low_cpu_mem_usage=True,
                max_memory=memory_map
            ).model
            # self.llm = AutoModelForCausalLM.from_pretrained(
            #     llm_path, 
            #     device_map='cpu',
            #     low_cpu_mem_usage=True,
            # )
            
            # Iterate over named buffers
            import torch.nn as nn
            buffer_keywords = ["qweight", "qzeros", "scales"]
            for name, buffer in list(self.llm.named_buffers()):  # Use list() to avoid modification issues during iteration
                if any(keyword in name for keyword in buffer_keywords):
                    # Extract the parent module and attribute name
                    module_name, buffer_name = name.rsplit('.', 1)
                    parent_module = dict(self.llm.named_modules())[module_name]
                    
                    # Unregister the buffer
                    buffer_data = getattr(parent_module, buffer_name)
                    delattr(parent_module, buffer_name)  # Remove it from the module

                    # Register it as a trainable parameter
                    parent_module.register_parameter(buffer_name, nn.Parameter(buffer_data, requires_grad=False))

        else: 
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path, 
                device_map="cpu", 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.float16
            )

        estimated_mem = torch.cuda.memory_allocated(device)
        logging.info(f"Init Allocated Memory = {estimated_mem / (1024 ** 3)} GB")
        
        for param in self.llm.model.embed_tokens.parameters():
            estimated_mem += param.numel() * param.element_size()
        for param in self.llm.lm_head.parameters():
            estimated_mem += param.numel() * param.element_size()
        estimated_mem = estimated_mem / (1024 ** 3)
        
        decoder_layer_mem = 0.0
        for param in self.llm.model.layers[0].parameters():
            decoder_layer_mem += param.numel() * param.element_size()
        for buffer in self.llm.model.layers[0].buffers():
            decoder_layer_mem += buffer.numel() * buffer.element_size()
            
        decoder_layer_mem = decoder_layer_mem / (1024 ** 3)
        memory_limit = memory_limit / 1.2

        # TODO: Check the memory usage to check how much layers to be offloaded
        for i in range(len(self.llm.model.layers)):
            # if estimated_mem <= memory_limit - 2 * decoder_layer_mem - 0.5:
            if estimated_mem <= memory_limit - decoder_layer_mem:
                estimated_mem += decoder_layer_mem
                device_map[f"model.layers.{i}"] = device
            else:
                device_map[f"model.layers.{i}"] = "cpu"
        logging.info(f"[Check] device_map: {device_map}")
        
        # set pin_memory to reduce memory access time
        for layer in self.llm.model.layers:
            for param in layer.parameters():
                param.data = param.data.cpu().pin_memory(device)
        estimated_mem = torch.cuda.memory_allocated(device)

        self.llm = dispatch_model(self.llm, device_map=device_map)
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
        logging.info(f"Allocated Memory = {allocated_memory} GB")
        
        if allocated_memory > memory_limit:
            logging.info(f"[Warning] memory usage is too much")

        
    def _speculate(self, input_ids, hidden_states, past_key_values):
        return super()._speculate(input_ids=input_ids, hidden_states=hidden_states, past_key_values=past_key_values)

    def _tree_decoding(self, root, past_key_values, position_offset, device, dtype=torch.float32):
        return super()._tree_decoding(root, past_key_values, position_offset, device, dtype=dtype)

    def _generate(
        self, 
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_warper: LogitsWarper,
        do_sample: bool,
    ):   
        return super()._generate(input_ids, stopping_criteria, logits_warper, do_sample)             

class ProfileOffloadSDWrapper(OffloadSDWrapper):
    def __init__(self, method="greedy", out_dir="specdecodes/experiments/profile_data", prefix="sd"):
        super(ProfileOffloadSDWrapper, self).__init__(method=method)
        self.profile_data = {}
        self.sampled_count = 1 # assume first token is sampled (prefill stage)
        self.iter_count = 1 # assume first step is done (prefill stage)
        
        self.out_dir = out_dir
        self.prefix = prefix

        self.exp_log = {}
        self.draft_time_per_iter = []
        self.target_time_per_iter = []
        self.verify_time_per_iter = []
        
    
    def _speculate(self, input_ids, hidden_states, past_key_values):
        start_time = time.perf_counter()
        # with record_function("_speculate"):
        #     root = super()._speculate(input_ids, hidden_states, past_key_values)
        root = super()._speculate(input_ids, hidden_states, past_key_values)
        self.draft_time_per_iter.append(time.perf_counter()-start_time)
        return root
    
    def _tree_decoding(self, root, past_key_values, position_offset, device, dtype=torch.float32):
        start_time = time.perf_counter()
        # with record_function("_tree_decoding"):
        #     outputs = super()._tree_decoding(root, past_key_values, position_offset, device, dtype)
        outputs = super()._tree_decoding(root, past_key_values, position_offset, device, dtype)
        self.target_time_per_iter.append(time.perf_counter()-start_time)
        return outputs
    
    def _verify(self, root, logits, logits_warper, do_sample):
        start_time = time.perf_counter()
        # with record_function("_verify"):
        #     sampled_tokens, hidden_indices, (total_len, accept_len) = super()._verify(root, logits, logits_warper, do_sample)
        sampled_tokens, hidden_indices, (total_len, accept_len) = super()._verify(root, logits, logits_warper, do_sample)
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
        org_input_len = len(input_ids[0])
        start_time = time.perf_counter()
        input_ids = super()._generate(input_ids, stopping_criteria, logits_warper, do_sample)
        end_time = time.perf_counter()
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     input_ids = super()._generate(input_ids, stopping_criteria, logits_warper, do_sample)
        # end_time = time.perf_counter()

        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        # prof.export_chrome_trace("trace.json")
        
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