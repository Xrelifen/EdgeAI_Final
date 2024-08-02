import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache

from bigtree import levelorder_iter

def invert_mask(mask, dtype):
    inverted_mask = ~mask
    mask = inverted_mask.to(dtype=dtype).masked_fill(
        inverted_mask, torch.finfo(dtype).min
    )
    return mask

def build_tree_attention_data(root, position_offset, dtype):
    candidate_nodes = list(levelorder_iter(root))
    candidate_len = len(list(candidate_nodes))
    # label nodes with index
    for idx, node in enumerate(candidate_nodes):
        node.ind = idx
    
    # build tree mask
    tree_mask = torch.zeros((candidate_len, candidate_len), dtype=torch.bool)
    for idx, node in enumerate(candidate_nodes):
        if node.parent is not None:
            tree_mask[idx, :] = tree_mask[node.parent.ind, :]
        tree_mask[idx, idx] = True
    tree_mask = tree_mask[None][None]
    tree_mask = torch.concat((torch.ones([1, 1, candidate_len, position_offset], dtype=torch.bool), tree_mask), dim=3)
    tree_mask = invert_mask(tree_mask, dtype)

    # tree_candidates = node id
    tree_input_ids = torch.tensor([node.id for node in candidate_nodes], dtype=torch.long)[None]
    
    # tree_position_id = node depth + offset
    tree_position_ids = torch.tensor([(position_offset + node.depth - 1) for node in candidate_nodes], dtype=torch.long)[None]
    
    return tree_input_ids, tree_position_ids, tree_mask

class TreeDynamicCache(DynamicCache):
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