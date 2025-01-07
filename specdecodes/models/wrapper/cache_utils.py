from typing import Optional
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.configuration_utils import PretrainedConfig
from typing import Dict, Union

def invert_mask(mask, dtype): 
    # Inversion using bitwise NOT and multiplication
    return (~mask).to(dtype) * torch.finfo(dtype).min

class TreeDynamicCache(DynamicCache):
    def __init__(self) -> None:
        super().__init__()
        
    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

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
            
    def reorder_cache_with_offset(self, beam_idx: torch.LongTensor, new_chunk_len=1, offset=0, dim=0):
        """Reorders the cache for beam search, given the selected beam indices, while [:offset] remain unchanged""" 
        # Create the full beam index by concatenating the unchanged range with the adjusted beam indices
        beam_idx_full = torch.cat(
            [torch.arange(offset, device=beam_idx.device), beam_idx + offset],
            dim=0
        )

        # Dictionary to cache beam_idx tensors per device to avoid redundant .to(device) calls
        beam_idx_device_cache = {}

        # Iterate over key_cache and value_cache simultaneously
        for layer_idx, (key_layer, value_layer) in enumerate(zip(self.key_cache, self.value_cache)):
            # Determine the device of the current layers
            device = key_layer.device  # Assuming key_cache and value_cache layers are on the same device

            # Retrieve or create the cached beam indices for the current device
            if device not in beam_idx_device_cache:
                beam_idx_device_cache[device] = beam_idx_full.to(device)

            reordered_idx = beam_idx_device_cache[device]

            # Reorder key_cache and value_cache by assigning the indexed tensors back
            self.key_cache[layer_idx] = key_layer.index_select(dim, reordered_idx)
            self.value_cache[layer_idx] = value_layer.index_select(dim, reordered_idx)
            
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
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search.
        
        Args:
            max_length (int): The new maximum sequence length. Can be negative to remove tokens from the end.
        """
        seq_length = self.get_seq_length()
        
        # Handle negative max_length by removing tokens from the end
        if max_length < 0:
            max_length = seq_length - abs(max_length)

        # If current sequence length is within the desired max_length, no action is needed
        if seq_length <= max_length:
            return

        # Determine the device type once to avoid repetitive checks inside the loop
        device = self.key_cache[0].device
        device_type = device.type

        if device_type != 'mps':
            # Use index_fill_ for non-MPS devices
            index = torch.arange(max_length, seq_length, device=device)
            for key, value in zip(self.key_cache, self.value_cache):
                key.index_fill_(dim=2, index=index, value=0)
                value.index_fill_(dim=2, index=index, value=0)
        else:
            # For MPS devices, use slicing assignments which are more efficient
            # and avoid the NotImplementedError associated with index_fill_
            for key, value in zip(self.key_cache, self.value_cache):
                key[:, :, max_length:] = 0
                value[:, :, max_length:] = 0
            
    def reorder_cache_with_offset(self, beam_idx: torch.LongTensor, new_chunk_len=1, offset=0, dim=0):
        """
        Reorder the newly added cache slice [offset : offset + total_new_len]
        according to beam_idx (of length slice_len), then zero out the leftover
        [offset + slice_len : offset + total_new_len].
        """
        slice_len = beam_idx.size(0)
        leftover_len = new_chunk_len - slice_len
        beam_idx_device_cache = {}

        for k, v in zip(self.key_cache, self.value_cache):
            # Cache the beam_idx tensor on the current device (minimizes .to(device) calls)
            dev = k.device
            if dev not in beam_idx_device_cache:
                beam_idx_device_cache[dev] = beam_idx.to(dev)
            b_idx = beam_idx_device_cache[dev]

            # Narrow to the newly added slice, reorder it by b_idx, and copy to the first slice_len
            slice_k = k.narrow(dim, offset, new_chunk_len)
            slice_v = v.narrow(dim, offset, new_chunk_len)

            k.narrow(dim, offset, slice_len).copy_(slice_k.index_select(dim, b_idx))
            v.narrow(dim, offset, slice_len).copy_(slice_v.index_select(dim, b_idx))

            # Zero out the leftover portion
            if leftover_len > 0:
                k.narrow(dim, offset + slice_len, leftover_len).zero_()
                v.narrow(dim, offset + slice_len, leftover_len).zero_()