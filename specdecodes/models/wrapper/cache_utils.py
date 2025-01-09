from typing import Optional, Dict, Union
import nvtx
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.configuration_utils import PretrainedConfig

class TreeDynamicCache(DynamicCache):
    def __init__(self) -> None:
        super().__init__()
        
    def crop(self, max_length: int):
        """Crop the past key/values up to a new `max_length` (negative removes from the end)."""
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for i in range(len(self.key_cache)):
            if self.key_cache[i] != []:
                self.key_cache[i] = self.key_cache[i][..., :max_length, :]
                self.value_cache[i] = self.value_cache[i][..., :max_length, :]
                
    def reorder_cache(self, beam_idx: torch.LongTensor, dim=0):
        """Reorder cache for beam search (classic approach)."""
        for i in range(len(self.key_cache)):
            dev = self.key_cache[i].device
            self.key_cache[i] = self.key_cache[i].index_select(dim, beam_idx.to(dev))
            self.value_cache[i] = self.value_cache[i].index_select(dim, beam_idx.to(dev))
            
    def reorder_cache_with_offset(self, beam_idx: torch.LongTensor, new_chunk_len=1, offset=0, dim=0):
        """
        Reorder the cache for beam search with an offset. 
        [:offset] remain unchanged; [offset:] is reordered.
        """
        # Build the full reorder indices
        full_beam_idx = torch.cat(
            [torch.arange(offset, device=beam_idx.device), beam_idx + offset], dim=0
        )
        beam_idx_device_cache = {}

        for i in range(len(self.key_cache)):
            dev = self.key_cache[i].device
            if dev not in beam_idx_device_cache:
                beam_idx_device_cache[dev] = full_beam_idx.to(dev)
            r_idx = beam_idx_device_cache[dev]
            
            self.key_cache[i] = self.key_cache[i].index_select(dim, r_idx)
            self.value_cache[i] = self.value_cache[i].index_select(dim, r_idx)


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
        Crop past key/values up to `max_length` (negative removes from the end).
        Sets leftover tokens to zero for non-MPS devices using index_fill_.
        """
        seq_length = self.get_seq_length()
        if max_length < 0:
            max_length = seq_length - abs(max_length)
        if seq_length <= max_length:
            return

        dev = self.key_cache[0].device
        if dev.type != 'mps':
            idx = torch.arange(max_length, seq_length, device=dev)
            for k, v in zip(self.key_cache, self.value_cache):
                k.index_fill_(dim=2, index=idx, value=0)
                v.index_fill_(dim=2, index=idx, value=0)
        else:
            for k, v in zip(self.key_cache, self.value_cache):
                k[:, :, max_length:] = 0
                v[:, :, max_length:] = 0

    def naive_reorder_cache_with_offset(self, beam_idx: torch.LongTensor, new_chunk_len=1, offset=0, dim=0):
        """Straightforward reorder of [offset : offset + new_chunk_len] with leftover zeroing."""
        slice_len = beam_idx.size(0)
        leftover_len = new_chunk_len - slice_len
        beam_idx_device_cache = {}

        for k, v in zip(self.key_cache, self.value_cache):
            dev = k.device
            if dev not in beam_idx_device_cache:
                beam_idx_device_cache[dev] = beam_idx.to(dev)
            b_idx = beam_idx_device_cache[dev]
            
            reorder_indices = offset + b_idx
            r_k = k.index_select(dim, reorder_indices)
            r_v = v.index_select(dim, reorder_indices)
            k.narrow(dim, offset, slice_len).copy_(r_k)
            v.narrow(dim, offset, slice_len).copy_(r_v)
            if leftover_len > 0:
                k.narrow(dim, offset + slice_len, leftover_len).zero_()
                v.narrow(dim, offset + slice_len, leftover_len).zero_()
          
    def reorder_cache_with_offset(
        self,
        beam_idx: torch.LongTensor,
        new_chunk_len: int = 1,
        offset: int = 0,
        dim: int = 0,
    ):
        """
        Reorder [offset : offset + new_chunk_len] in the key/value cache using beam_idx,
        then zero leftover. Batches layers by device and uses in-place ops for efficiency.
        """
        slice_len = beam_idx.size(0)
        leftover_len = new_chunk_len - slice_len

        # Group layers by device
        with nvtx.annotate("group by device", color="green"):
            device_layers = {}
            for i, (k, v) in enumerate(zip(self.key_cache, self.value_cache)):
                dev = k.device
                device_layers.setdefault(dev, {"k_list": [], "v_list": [], "idx": []})
                device_layers[dev]["k_list"].append(k)
                device_layers[dev]["v_list"].append(v)
                device_layers[dev]["idx"].append(i)

        # Transfer beam_idx to each device non-blocking
        with nvtx.annotate("beam_idx transfer", color="blue"):
            beam_idx_per_device = {
                dev: beam_idx.to(dev, non_blocking=True) for dev in device_layers
            }
            torch.cuda.synchronize()

        # In-place reorder + zero leftover per device
        for dev, data in device_layers.items():
            with nvtx.annotate(f"reorder device={dev}", color="purple"):
                b_idx = beam_idx_per_device[dev]
                k_cat = torch.stack(data["k_list"], dim=0)
                v_cat = torch.stack(data["v_list"], dim=0)

                # Build in-place reorder indices
                reorder_dest = offset + torch.arange(slice_len, device=dev)
                reorder_src = offset + b_idx

                # Reorder slice in-place
                k_cat.index_copy_(dim + 1, reorder_dest, k_cat.index_select(dim + 1, reorder_src))
                v_cat.index_copy_(dim + 1, reorder_dest, v_cat.index_select(dim + 1, reorder_src))

                # Zero leftover
                if leftover_len > 0:
                    leftover_range = offset + torch.arange(slice_len, new_chunk_len, device=dev)
                    k_cat.index_fill_(dim + 1, leftover_range, 0)
                    v_cat.index_fill_(dim + 1, leftover_range, 0)

                # Split back into per-layer Tensors
                unbound_k = torch.unbind(k_cat, dim=0)
                unbound_v = torch.unbind(v_cat, dim=0)
                for i_layer, new_k, new_v in zip(data["idx"], unbound_k, unbound_v):
                    self.key_cache[i_layer] = new_k
                    self.value_cache[i_layer] = new_v