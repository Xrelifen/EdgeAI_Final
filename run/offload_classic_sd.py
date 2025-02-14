from .app_router import run_app
from .base import BaseBuilder

import torch
from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.draft_models.classic_sd import ClassicSDDraftModel
from specdecodes.models.generators.classic_sd import ClassicSDGenerator

from hqq.core.quantize import *

from transformers import AutoTokenizer, AutoModelForCausalLM
from specdecodes.models.utils.modeling_utils import get_named_tensors

import logging

def temp_recipe(model, draft_model, vram_limit):  
    # Quantization
    base_quant_config_a = BaseQuantizeConfig(nbits=4, group_size=32, axis=1)
    
    quant_config = {}
    for i in range(0, 31+1):
        quant_config[f"layers.{i}.self_attn.q_proj"] = base_quant_config_a
        quant_config[f"layers.{i}.self_attn.k_proj"] = base_quant_config_a
        quant_config[f"layers.{i}.self_attn.v_proj"] = base_quant_config_a
        quant_config[f"layers.{i}.self_attn.o_proj"] = base_quant_config_a
        quant_config[f"layers.{i}.mlp.gate_proj"] = base_quant_config_a
        quant_config[f"layers.{i}.mlp.up_proj"] = base_quant_config_a 
        quant_config[f"layers.{i}.mlp.down_proj"] = base_quant_config_a

    target_config = None
    draft_config = {
        "quant_config": {
            "config": quant_config,
            "backend": "gemlite",
        },
    }
    
    return target_config, draft_config

class OffloadClassicSDBuilder(BaseBuilder):
    def __init__(self):
        super().__init__()
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.draft_model_path = "meta-llama/Llama-3.2-1B-Instruct"
        
        # Generator configurations
        self.generator_class = ClassicSDGenerator
        self.draft_params = DraftParams(
            max_depth=12,
            topk_len=16,
            max_verify_tokens=192,
            min_accept_prob=1e-8,
        )
        
        # Offloading
        # self.recipe = temp_recipe
        self.vram_limit = 8.0 # in GB
        
        # Speed up inference using torch.compile
        # self.cache_implementation = "static"
        # self.warmup_iter = 10
        # self.compile_mode = "max-autotune"
        
        # Profiling
        self.generator_profiling = True
        self.nvtx_profiling = False

    def _load_model_and_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
            _attn_implementation="sdpa",
        )

        model_config = model.config
        for layer in model.layers:
            for param in layer.parameters():
                param.data = param.data.cpu().pin_memory(self.device)
            for buffer in layer.buffers():
                buffer.data = buffer.data.cpu().pin_memory(self.device)

        # TODO: Does .to(device) actually work
        # Set rotary_emb & rmsnorm to device
        for _, tensor in get_named_tensors(model.rotary_emb):
            tensor.data = tensor.data.to(self.device)
        for _, tensor in get_named_tensors(model.norm):
            tensor.data = tensor.data.to(self.device)

        # Set embed_tokens and lm_head to device
        for _, tensor in get_named_tensors(model.get_input_embeddings()):
            tensor.data = tensor.data.to(self.device)
        for _, tensor in get_named_tensors(model.get_output_embeddings()):
            tensor.data = tensor.data.to(self.device)

        mem_usage = torch.cuda.memory_allocated(self.device) / (10 ** 9)
        est_mem_usage = mem_usage / 0.9
        logging.info(f"Init Allocated Memory = {est_mem_usage} GB")
        
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
        est_mem_usage += self.draft_params.max_verify_tokens * model_config.vocab_size * 2 * 2 / (10 ** 9)
        head_dim = model_config.hidden_size / model_config.num_attention_heads
        kv_dim = head_dim * model_config.num_key_value_heads
        est_mem_usage += self.max_length * kv_dim * 2 * model_config.num_hidden_layers * 2 * 2 / (10 ** 9)

        prefetch_name_map = {}
        module_map = {}
        device_map = {}
        for block_n in range(model_config.num_hidden_layers):
            for layer_n in range(len(llama_layer)):
                layer_name = llama_layer[layer_n]
                prefixed_layer_name = f'{block_n}.{layer_name}'
                
                next_layer_n = (layer_n+1) % len(llama_layer)
                if next_layer_n < layer_n:
                    prefixed_next_layer_name = f'{block_n+1}.{llama_layer[next_layer_n]}'
                else:
                    prefixed_next_layer_name = f'{block_n}.{llama_layer[next_layer_n]}'

                if est_mem_usage <= self.vram_limit - 3 * max_mem:
                    device_map[prefixed_layer_name] = self.device
                    est_mem_usage += llama_layer_mem[layer_name]
                    if est_mem_usage >= self.vram_limit - 3 * max_mem:
                        prefetch_name_map[prefixed_layer_name] = prefixed_next_layer_name
                else:
                    device_map[prefixed_layer_name] = 'cpu'
                    if block_n != model_config.num_hidden_layers - 1 or layer_n != len(llama_layer) - 1:
                        prefetch_name_map[prefixed_layer_name] = prefixed_next_layer_name

                layer_name_split = layer_name.split('.')
                module = self.llm.model.layers[block_n]
                for sublayer_name in layer_name_split:
                    module = getattr(module, sublayer_name, None)
                module_map[prefixed_layer_name] = module
                assert module_map[prefixed_layer_name] is not None, "module not found"
        
        logging.info(f'[Estimated Memory Usage] {est_mem_usage} GB')
        logging.info(f'[Check Device Map]')
        for module_name, dev in device_map.items():
            logging.info(f'{module_name}: {dev}')
        logging.info(f'[Check Prefetch Map]')
        for module_name, next_module_name in prefetch_name_map.items():
            logging.info(f'{module_name} - {next_module_name}')
        
        # TODO: prefetch next layer
        # if 'autoawq' in llm_path:
        #     offload_buffers = ["qweight", "qzeros", "scales"]
        #     self.llm = dispatch_model(self.llm, device_map=device_map, offload_buffers=offload_buffers)
        # else:
        #     self.llm.model.layers = dispatch_model_with_prefetch(
        #         self.llm.model.layers,
        #         device_map=device_map,
        #         prefetch_name_map=prefetch_name_map,
        #         module_map=module_map
        #     )

        allocated_memory = (torch.cuda.memory_allocated(self.device) / 0.9) / (10 ** 9)
        logging.info(f"[Memory After Dispatch] {allocated_memory} GB")
        
        if allocated_memory > self.vram_limit:
            logging.info(f"[Warning] memory usage is too much")

        return model, tokenizer
    
    def _load_draft_model(self, target_model=None, tokenizer=None, draft_path=None):
        draft_model = ClassicSDDraftModel.from_pretrained(
            draft_path,
            target_model=target_model,
            torch_dtype=self.dtype,
            eos_token_id=tokenizer.eos_token_id
        ).to(self.device)
        draft_model.update_modules(embed_tokens=target_model.get_input_embeddings(), lm_head=target_model.lm_head)
        return draft_model
    
    
if __name__ == "__main__":
    run_app(OffloadClassicSDBuilder())