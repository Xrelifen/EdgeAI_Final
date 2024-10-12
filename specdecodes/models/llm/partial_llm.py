from copy import deepcopy
import torch
import torch.nn as nn
from . import modeling_llama

class LLM_First_Layers(nn.Module):
    def __init__(self, llm, keep_layers_num=1):
        super().__init__()
        begin_idx = 0
        config = deepcopy(llm.config)
        config.org_num_hidden_layers = config.num_hidden_layers
        config.num_hidden_layers = keep_layers_num
        self.keep_layers_num = keep_layers_num
        self.config = config
        
        # copy the embed_tokens
        embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        embed_tokens.weight.data = llm.get_input_embeddings().weight.data
        self.embed_tokens = embed_tokens
        
        # copy the last keep_layers_num layers
        if keep_layers_num > 0:
            model = modeling_llama.LlamaModel(config)
            model.norm.weight.data = llm.model.norm.weight.data
            for idx, decoder_layer in enumerate(model.layers):
                real_idx = begin_idx + idx
                decoder_layer.self_attn.layer_idx = real_idx # assigning correct layer index for kv_cache
                for param, llm_param in zip(decoder_layer.parameters(), llm.model.layers[real_idx].parameters()):
                    param.data = llm_param.data
            del model.embed_tokens
            self.model = model

        # defaults set all requires_grad to False
        self.freeze()
        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
            
    @property
    def dtype(self):
        return self.lm_head.weight.dtype
        
    def forward(self, input_ids, embed_only=False, output_last_hidden_states=False, *model_args, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        
        # compute hidden states
        if not embed_only and self.keep_layers_num > 0:
            outputs = self.model(inputs_embeds=hidden_states, *model_args, **kwargs)
            hidden_states = outputs.last_hidden_state
        
        # prepare output
        output = (hidden_states, )
        if output_last_hidden_states:
            output = output + (hidden_states,)
        
        return output if len(output) > 1 else output[0]

class LLM_Last_Layers(nn.Module):
    def __init__(self, llm, keep_layers_num=1):
        super().__init__()
        begin_idx = llm.config.num_hidden_layers - keep_layers_num
        config = deepcopy(llm.config)
        config.org_num_hidden_layers = config.num_hidden_layers
        config.num_hidden_layers = keep_layers_num
        self.keep_layers_num = keep_layers_num
        self.config = config
        
        # copy the lm_head
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        lm_head.weight.data = llm.lm_head.weight.data
        self.lm_head = lm_head
        
        # copy the last keep_layers_num layers
        if keep_layers_num > 0:
            model = modeling_llama.LlamaModel(config)
            model.norm.weight.data = llm.model.norm.weight.data
            for idx, decoder_layer in enumerate(model.layers):
                real_idx = begin_idx + idx
                decoder_layer.self_attn.layer_idx = real_idx # assigning correct layer index for kv_cache
                for param, llm_param in zip(decoder_layer.parameters(), llm.model.layers[real_idx].parameters()):
                    param.data = llm_param.data
            del model.embed_tokens
            self.model = model

        # defaults set all requires_grad to False
        self.freeze()
        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
            
    @property
    def dtype(self):
        return self.lm_head.weight.dtype
        
    def forward(self, hidden_states, head_only=False, output_last_hidden_states=False, *model_args, **kwargs):
        # compute hidden states
        if not head_only and self.keep_layers_num > 0:
            outputs = self.model(inputs_embeds=hidden_states, *model_args, **kwargs)
            hidden_states = outputs.last_hidden_state

        # compute logits
        logits = self.lm_head(hidden_states)
        
        # prepare output
        output = (logits,)
        if output_last_hidden_states:
            output = output + (hidden_states,)
        
        return output if len(output) > 1 else output[0]