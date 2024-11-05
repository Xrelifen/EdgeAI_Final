from typing import Optional
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache

from bigtree import levelorder_iter

def invert_mask(mask, dtype): 
    # Inversion using bitwise NOT and multiplication
    return (~mask).to(dtype) * torch.finfo(dtype).min

def keep_top_n_nodes(root, n):
    if n == -1:
        return root
    
    candidate_nodes = list(levelorder_iter(root))
    n = min(n, len(candidate_nodes))
    _, topk_indices = torch.topk(torch.tensor([node.global_prob for node in candidate_nodes]), n)
    topk_nodes = [candidate_nodes[idx] for idx in topk_indices]
    
    # Assign indices to candidate nodes
    for idx, node in enumerate(topk_nodes):
        node.ind = idx
    
    return root

def build_tree_attention_data(root, position_offset, dtype):
    candidate_nodes = list(levelorder_iter(root))
    candidate_len = len(candidate_nodes)
    
    # Assign indices to candidate nodes 
    for idx, node in enumerate(candidate_nodes):
        node.ind = idx

    # Initialize tree mask
    tree_mask = torch.zeros((candidate_len, candidate_len), dtype=torch.bool)

    # Set mask entries using ancestor indices
    for node in candidate_nodes:
        ancestor_indices = []
        current_node = node
        while current_node:
            ancestor_indices.append(current_node.ind)
            current_node = current_node.parent
        tree_mask[node.ind, ancestor_indices] = True

    # Append offset mask for previous tokens
    offset_mask = torch.ones((candidate_len, position_offset), dtype=torch.bool)
    tree_mask = torch.cat((offset_mask, tree_mask), dim=1)

    # Invert the mask
    tree_mask = invert_mask(tree_mask, dtype)
    tree_mask = tree_mask.unsqueeze(0).unsqueeze(0)

    # Prepare input and position IDs
    tree_input_ids = torch.tensor([node.id for node in candidate_nodes], dtype=torch.long).unsqueeze(0)
    tree_position_ids = torch.tensor([position_offset + node.depth - 1 for node in candidate_nodes], dtype=torch.long).unsqueeze(0)

    return tree_input_ids, tree_position_ids, tree_mask

def make_tree_attention_mask(
        prefix_len: int,
        gen_len: int,
        ancestors: list[list[int]],
        device="cpu",
        dtype=torch.float32
    ) -> torch.FloatTensor:
    mask = torch.zeros((gen_len, gen_len + prefix_len), dtype=torch.bool, device=device)

    # Set mask by using advanced indexing
    row_indices = []
    col_indices = []
    for idx, ancestor_list in enumerate(ancestors):
        if ancestor_list:
            row_indices.extend([idx] * len(ancestor_list))
            col_indices.extend(ancestor_list)

    if row_indices:
        mask[row_indices, col_indices] = True

    # Invert the mask
    mask = invert_mask(mask, dtype)

    return mask.unsqueeze(0).unsqueeze(0)

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