from typing import Optional
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.configuration_utils import PretrainedConfig
from typing import Any, Dict, List, Optional, Tuple, Union

def invert_mask(mask, dtype): 
    # Inversion using bitwise NOT and multiplication
    return (~mask).to(dtype) * torch.finfo(dtype).min

class TreeDynamicCache(DynamicCache):
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        
    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        # if self.get_seq_length() <= max_length:
        #     return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx] != []:
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]
                
    def reorder_cache(self, beam_idx: torch.LongTensor, dim=0):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(dim, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(dim, beam_idx.to(device))

    def reorder_cache_with_offset(self, beam_idx: torch.LongTensor, offset=0, dim=0):
        """Reorders the cache for beam search, given the selected beam indices, while [:offset] remain unchanged""" 
        beam_idx = torch.cat([torch.arange(offset, device=beam_idx.device), beam_idx + offset], dim=0)
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(dim, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(dim, beam_idx.to(device))

class TreeStaticCache(StaticCache):
    def __init__(
        self,
        config: PretrainedConfig,
        max_cache_len: int = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: Optional[int] = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__(
            config=config,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
            max_batch_size=max_batch_size,
            layer_device_map=layer_device_map,
        )
        
    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        seq_length = self.get_seq_length()
        # In case it is negative
        if max_length < 0:
            max_length = seq_length - abs(max_length)

        if seq_length <= max_length:
            return

        begin = max_length
        end = seq_length + 1
        index = torch.arange(begin, end, device=self.key_cache[0].device)

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            try:
                self.key_cache[idx].index_fill_(2, index, 0)
                self.value_cache[idx].index_fill_(2, index, 0)
            except NotImplementedError:
                # The operator 'aten::index_fill' is not currently implemented for the MPS device.
                self.key_cache[idx][:, :, index] = 0
                self.value_cache[idx][:, :, index] = 0
                
    def reorder_cache_with_offset(self, beam_idx: torch.LongTensor, offset=0, dim=0):
        """Reorders the cache for beam search, given the selected beam indices, while [:offset] remain unchanged""" 
        beam_idx = torch.cat([torch.arange(offset, device=beam_idx.device), beam_idx + offset], dim=0)
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(dim, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(dim, beam_idx.to(device))