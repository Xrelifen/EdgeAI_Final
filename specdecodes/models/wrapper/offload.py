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

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.generation.logits_process import LogitsWarper
from transformers.generation.stopping_criteria import StoppingCriteria
from accelerate import (
    dispatch_model, 
    infer_auto_device_map
)
from accelerate.utils import named_module_tensors
from ..utils import (
    set_module_tensor_to_device,
    dispatch_model_with_prefetch,
    DraftParams, 
    invert_mask
)

import specdecodes.models.llm.modeling_llama as modeling_llama 

from transformers.cache_utils import StaticCache, DynamicCache

from bigtree import preorder_iter, levelorder_iter
from bigtree import tree_to_nested_dict

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
            self.llm = modeling_llama.LlamaForCausalLM.from_pretrained(
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
    def __init__(self, draft_params: DraftParams, *model_args, **kwargs):
        super(OffloadSDWrapper, self).__init__(draft_params, *model_args, **kwargs)

    def set_offload_llm(self, llm_path, memory_limit=6.0, device="cuda:0"):
        assert self.ssm is not None, "SSM model must first be loaded on gpu"
        # device_map = {
        #     "model.embed_tokens": "cuda:0",
        #     "model.norm": "cuda:0",
        #     "model.rotary_emb": "cuda:0",
        #     "lm_head": "cuda:0",
        # }
        logging.info(f'[Memory Limit]: {memory_limit} GB')
        if 'autoawq' in llm_path:
            memory_map = {0: "0GiB", "cpu": "99GiB"}
            self.llm = AutoAWQForCausalLM.from_quantized(
                llm_path, 
                fuse_layers=False,
                low_cpu_mem_usage=True,
                max_memory=memory_map
            ).model

        else: 
            self.llm = modeling_llama.LlamaForCausalLM.from_pretrained(
                llm_path, 
                device_map="cpu", 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.float16
            )

        llm_config = self.llm.config
        for layer in self.llm.model.layers:
            for param in layer.parameters():
                param.data = param.data.cpu().pin_memory(device)
            for buffer in layer.buffers():
                buffer.data = buffer.data.cpu().pin_memory(device)

        # Set rotary_emb & rmsnorm to device
        for tensor_name, _ in named_module_tensors(self.llm.model.rotary_emb):
            set_module_tensor_to_device(self.llm.model.rotary_emb, tensor_name, device)
        for tensor_name, _ in named_module_tensors(self.llm.model.norm):
            set_module_tensor_to_device(self.llm.model.norm, tensor_name, device)

        # Set embed_tokens and lm_head to device
        embed_tokens = self.llm.get_input_embeddings()
        for tensor_name, _ in named_module_tensors(embed_tokens):
            set_module_tensor_to_device(embed_tokens, tensor_name, device)
        lm_head = self.llm.get_output_embeddings()
        for tensor_name, _ in named_module_tensors(lm_head):
            set_module_tensor_to_device(lm_head, tensor_name, device)

        mem_usage = torch.cuda.memory_allocated(device) / (10 ** 9)
        estimated_mem_usage = mem_usage / 0.9
        logging.info(f"Init Allocated Memory = {estimated_mem_usage} GB")
        
        # Estimated Memory Usage for each sublayer
        llama_layer = ['input_layernorm', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'post_attention_layernorm', \
                        'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        
        max_mem = 0
        llama_layer_mem = {}
        for layer_name in llama_layer:
            layer_mem = 0
            layer_name_split = layer_name.split('.')

            module = self.llm.model.layers[0]
            for sublayer_name in layer_name_split:
                module = getattr(module, sublayer_name, None)
                assert module is not None, "Sub-Layer not found in current module"

            for param in module.parameters():
                layer_mem += param.numel() * param.element_size()
            for buffer in module.buffers():
                layer_mem += buffer.numel() * buffer.element_size()
            layer_mem = (layer_mem / 0.9) / (10 ** 9)
            llama_layer_mem[layer_name] = layer_mem
            max_mem = max(max_mem, layer_mem)
        
        logging.info(f'[Check Llama Layer Mem Usage] {llama_layer_mem}')
        # Estimated Memory Usage For Device Map
        estimated_mem_usage += 512 * llm_config.vocab_size * 2 * 2 / (10 ** 9)
        head_dim = llm_config.hidden_size / llm_config.num_attention_heads
        kv_dim = head_dim * llm_config.num_key_value_heads
        estimated_mem_usage += 1024 * kv_dim * 2 * llm_config.num_hidden_layers * 2 * 2 / (10 ** 9)

        prefetch_name_map = {}
        module_map = {}
        device_map = {}
        for block_n in range(llm_config.num_hidden_layers):
            for layer_n in range(len(llama_layer)):
                layer_name = llama_layer[layer_n]
                prefixed_layer_name = f'{block_n}.{layer_name}'
                
                next_layer_n = (layer_n+1) % len(llama_layer)
                if next_layer_n < layer_n:
                    prefixed_next_layer_name = f'{block_n+1}.{llama_layer[next_layer_n]}'
                else:
                    prefixed_next_layer_name = f'{block_n}.{llama_layer[next_layer_n]}'

                if estimated_mem_usage <= memory_limit - 3 * max_mem:
                    device_map[prefixed_layer_name] = device
                    estimated_mem_usage += llama_layer_mem[layer_name]
                    if estimated_mem_usage >= memory_limit - 3 * max_mem:
                        prefetch_name_map[prefixed_layer_name] = prefixed_next_layer_name
                else:
                    device_map[prefixed_layer_name] = 'cpu'
                    if block_n != llm_config.num_hidden_layers - 1 or layer_n != len(llama_layer) - 1:
                        prefetch_name_map[prefixed_layer_name] = prefixed_next_layer_name

                layer_name_split = layer_name.split('.')
                module = self.llm.model.layers[block_n]
                for sublayer_name in layer_name_split:
                    module = getattr(module, sublayer_name, None)
                module_map[prefixed_layer_name] = module
                assert module_map[prefixed_layer_name] is not None, "module not found"
        
        logging.info(f'[Estimated Memory Usage] {estimated_mem_usage} GB')
        logging.info(f'[Check Device Map]')
        for module_name, dev in device_map.items():
            logging.info(f'{module_name}: {dev}')
        logging.info(f'[Check Prefetch Map]')
        for module_name, next_module_name in prefetch_name_map.items():
            logging.info(f'{module_name} - {next_module_name}')
        
        # TODO: prefetch next layer
        if 'autoawq' in llm_path:
            offload_buffers = ["qweight", "qzeros", "scales"]
            self.llm = dispatch_model(self.llm, device_map=device_map, offload_buffers=offload_buffers)
        else:
            self.llm.model.layers = dispatch_model_with_prefetch(
                self.llm.model.layers,
                device_map=device_map,
                prefetch_name_map=prefetch_name_map,
                module_map=module_map
            )

        allocated_memory = (torch.cuda.memory_allocated(device) / 0.9) / (10 ** 9)
        logging.info(f"[Memory After Dispatch] {allocated_memory} GB")
        
        if allocated_memory > memory_limit:
            logging.info(f"[Warning] memory usage is too much")

        
    def _speculate(self, input_ids, hidden_states, past_key_values, max_cache_len=None):
        return super()._speculate(input_ids=input_ids, hidden_states=hidden_states, past_key_values=past_key_values, max_cache_len=max_cache_len)

    def _tree_decoding(self, tree, past_key_values, position_offset, cache_position, device):
        return super()._tree_decoding(tree=tree, past_key_values=past_key_values, position_offset=position_offset, cache_position=cache_position, device=device)

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_warper: LogitsWarper,
        do_sample: bool,
        use_static_tree_cache: bool = False,
    ): 
        return super()._generate(
            input_ids=input_ids, 
            stopping_criteria=stopping_criteria, 
            logits_warper=logits_warper, 
            do_sample=do_sample, 
            # user_static_tree_cache=use_static_tree_cache
        )             

class ProfileOffloadSDWrapper(SDWrapper):
    def __init__(self, out_dir="specdecodes/experiments/profile_data", prefix="sd", *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        self.out_dir = out_dir
        self.prefix = prefix
        
        self.profile_data = {}
        self.sampled_count = 1 # assume first token is sampled (prefill stage)
        self.iter_count = 1 # assume first step is done (prefill stage)
        
        self.exp_log = {}
        self.draft_events = []
        self.target_events = []
        self.verify_events = []
        
        self.disable_logging = False
        
    def _speculate(self, *model_args, **kwargs):
        if self.disable_logging:
            return super()._speculate(*model_args, **kwargs)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        root = super()._speculate(*model_args, **kwargs)
        end_event.record()
        
        self.draft_events.append((start_event, end_event))
        return root
    
    def _tree_decoding(self, *model_args, **kwargs):
        if self.disable_logging:
            return super()._tree_decoding(*model_args, **kwargs)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        outputs = super()._tree_decoding(*model_args, **kwargs)
        end_event.record()
        
        self.target_events.append((start_event, end_event))
        return outputs
    
    def _verify(self, tree, *model_args, **kwargs):
        if self.disable_logging:
            return super()._verify(tree, *model_args, **kwargs)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        sampled_tokens, hidden_indices, (total_len, accept_len) = super()._verify(tree, *model_args, **kwargs)
        end_event.record()
        
        self.verify_events.append((start_event, end_event))
        
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
    
    def compute_average_times(self):
        """
        Synchronize once at the end, then compute average
        draft and target times from the recorded CUDA events.
        """
        # Ensure all CUDA kernels are done
        torch.cuda.synchronize()

        # Compute total time for draft iterations
        draft_time_total_ms = 0.0
        for (start_event, end_event) in self.draft_events:
            draft_time_total_ms += start_event.elapsed_time(end_event)  # returns time in ms

        # Compute total time for target iterations
        target_time_total_ms = 0.0
        for (start_event, end_event) in self.target_events:
            target_time_total_ms += start_event.elapsed_time(end_event)
            
        # Compute total time for verify iterations
        verify_time_total_ms = 0.0
        for (start_event, end_event) in self.verify_events:
            verify_time_total_ms += start_event.elapsed_time(end_event)

        # Average times (in milliseconds)
        draft_avg_ms = draft_time_total_ms / max(len(self.draft_events), 1)
        target_avg_ms = target_time_total_ms / max(len(self.target_events), 1)
        verify_avg_ms = verify_time_total_ms / max(len(self.verify_events), 1)

        # Convert to seconds if you prefer
        draft_avg_s = draft_avg_ms / 1000.0
        target_avg_s = target_avg_ms / 1000.0
        verify_avg_s = verify_avg_ms / 1000.0

        return draft_avg_s, target_avg_s, verify_avg_s
    
    def _generate(self, input_ids: torch.LongTensor, *model_args, **kwargs):
        if self.disable_logging:
            return super()._generate(input_ids, *model_args, **kwargs)
        
        self.profile_data = {}
        self.sampled_count = 1 # assume first token is sampled (prefill stage)
        self.iter_count = 1 # assume first step is done (prefill stage)
        
        self.exp_log = {}
        self.draft_events = []
        self.target_events = []
        self.verify_events = []
        
        cur_time = time.strftime("%Y%m%d-%H%M%S")
        # prepare output directory
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            out_path = os.path.join(self.out_dir, f"{self.prefix}_{cur_time}.json")
        else:
            out_path = None
        
        # run generation
        org_input_len = len(input_ids[0])
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        input_ids = super()._generate(input_ids, *model_args, **kwargs)
        end_event.record()
        
        # Make sure all CUDA ops have finished before measuring
        torch.cuda.synchronize()
        
        # Elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_s = elapsed_time_ms / 1000.0
        
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
        avg_draft_s, avg_target_s, avg_verify_s = self.compute_average_times()
        self.exp_log['avg_draft_time'] = avg_draft_s
        self.exp_log['avg_target_time'] = avg_target_s
        self.exp_log['avg_verify_time'] = avg_verify_s
        
        self.exp_log['avg_sampled'] = avg_sampled
        self.exp_log['n_tokens'] = len(input_ids[0][org_input_len:])
        self.exp_log['tput'] = len(input_ids[0][org_input_len:]) / elapsed_time_s
        logging.info(
            f"Average draft time: {self.exp_log['avg_draft_time']:.4f},"\
            f"\tAverage target time: {self.exp_log['avg_target_time']:.4f},"\
            f"\tAverage verify time: {self.exp_log['avg_verify_time']:.4f}"
            f"\nGenerated {self.exp_log['n_tokens']} tokens in {elapsed_time_s:.2f}s, throughput: {self.exp_log['tput']:.2f} tokens/s"
        )
        
        return input_ids