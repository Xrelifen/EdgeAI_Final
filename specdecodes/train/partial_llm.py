from copy import deepcopy
import torch
import torch.nn as nn
from transformers.models.llama import modeling_llama

class BaseLLMModule(nn.Module):
    def __init__(self):
        super().__init__()

    def freeze(self):
        """Freeze all parameters in the module."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters in the module."""
        for param in self.parameters():
            param.requires_grad = True

    @property
    def dtype(self):
        """Return the data type of the module's parameters."""
        raise NotImplementedError("Subclasses should implement this method.")


class LLM_First_Layers(BaseLLMModule):
    def __init__(self, llm, keep_layers_num=1):
        super().__init__()
        begin_idx = 0
        config = deepcopy(llm.config)
        config.org_num_hidden_layers = config.num_hidden_layers
        config.num_hidden_layers = keep_layers_num
        self.keep_layers_num = keep_layers_num
        self.config = config

        # Copy the embedding layer
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.embed_tokens.weight = nn.Parameter(
            llm.get_input_embeddings().weight.clone()
        )

        # Copy the first `keep_layers_num` layers
        if keep_layers_num > 0:
            model = modeling_llama.LlamaModel(config)
            model.norm.weight = nn.Parameter(llm.model.norm.weight.clone())
            for idx, decoder_layer in enumerate(model.layers):
                real_idx = begin_idx + idx
                decoder_layer.self_attn.layer_idx = real_idx  # Correct layer index for kv_cache
                for param, llm_param in zip(
                    decoder_layer.parameters(),
                    llm.model.layers[real_idx].parameters(),
                ):
                    param.data = llm_param.data.clone()
            del model.embed_tokens
            self.model = model

        # Freeze parameters by default
        self.freeze()

    @property
    def dtype(self):
        return self.embed_tokens.weight.dtype

    def forward(
        self,
        input_ids,
        embed_only=False,
        output_last_hidden_states=False,
        *model_args,
        **kwargs
    ):
        hidden_states = self.embed_tokens(input_ids)

        # Compute hidden states through the layers if required
        if not embed_only and self.keep_layers_num > 0:
            outputs = self.model(
                inputs_embeds=hidden_states, *model_args, **kwargs
            )
            hidden_states = outputs.last_hidden_state

        # Prepare output
        output = (hidden_states,)
        if output_last_hidden_states:
            output += (hidden_states,)

        return output if len(output) > 1 else output[0]


class LLM_Last_Layers(BaseLLMModule):
    def __init__(self, llm, keep_layers_num=1):
        super().__init__()
        begin_idx = llm.config.num_hidden_layers - keep_layers_num
        config = deepcopy(llm.config)
        config.org_num_hidden_layers = config.num_hidden_layers
        config.num_hidden_layers = keep_layers_num
        self.keep_layers_num = keep_layers_num
        self.config = config

        # Copy the language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(llm.lm_head.weight.clone())

        # Copy the last `keep_layers_num` layers
        if keep_layers_num > 0:
            model = modeling_llama.LlamaModel(config)
            model.norm.weight = nn.Parameter(llm.model.norm.weight.clone())
            for idx, decoder_layer in enumerate(model.layers):
                real_idx = begin_idx + idx
                decoder_layer.self_attn.layer_idx = real_idx  # Correct layer index for kv_cache
                for param, llm_param in zip(
                    decoder_layer.parameters(),
                    llm.model.layers[real_idx].parameters(),
                ):
                    param.data = llm_param.data.clone()
            del model.embed_tokens
            self.model = model

        # Freeze parameters by default
        self.freeze()

    @property
    def dtype(self):
        return self.lm_head.weight.dtype

    def forward(
        self,
        hidden_states,
        head_only=False,
        output_last_hidden_states=False,
        *model_args,
        **kwargs
    ):
        # Compute hidden states through the layers if required
        if not head_only and self.keep_layers_num > 0:
            outputs = self.model(
                inputs_embeds=hidden_states, *model_args, **kwargs
            )
            hidden_states = outputs.last_hidden_state

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Prepare output
        output = (logits,)
        if output_last_hidden_states:
            output += (hidden_states,)

        return output if len(output) > 1 else output[0]
