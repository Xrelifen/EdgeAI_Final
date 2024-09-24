import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache

from bigtree import levelorder_iter


def invert_mask(mask, dtype): 
    # assume input mask dtype is boolean
    mask = mask.to(dtype=dtype) # torch.float16, torch.float32 support
    mask = (1.0 - mask) * torch.finfo(dtype).min # 0.0 -> -inf, 1.0 -> 0.0
    
    return mask


def build_tree_attention_data(root, position_offset, dtype):
    # Build the list of candidate nodes
    candidate_nodes = list(levelorder_iter(root))
    candidate_len = len(candidate_nodes)
    
    # update node.ind for each node
    for idx, node in enumerate(candidate_nodes):
        node.ind = idx
    
    # Build the tree mask
    tree_mask = torch.zeros((candidate_len, candidate_len), dtype=torch.bool)
    
    # Each child node has the same mask as its parent, with itself set to True
    for idx, node in enumerate(candidate_nodes):
        if node.parent is not None:
            tree_mask[idx, :] = tree_mask[node.parent.ind, :]
        tree_mask[idx, idx] = True
    
    # Append mask for previous tokens, which should be all filled with ones.
    offset_mask = torch.ones([candidate_len, position_offset], dtype=torch.bool)
    tree_mask = torch.concat((offset_mask, tree_mask), dim=1)
    
    # Invert the mask
    tree_mask = invert_mask(tree_mask, dtype)
    
    # Reshape the mask to (1, 1, candidate_len, candidate_len) to match required shape
    tree_mask = tree_mask.unsqueeze(0).unsqueeze(0)

    # tree_candidates = node id
    tree_input_ids = torch.tensor([node.id for node in candidate_nodes], dtype=torch.long).unsqueeze(0)
    
    # tree_position_id = node depth + offset
    tree_position_ids = torch.tensor([(position_offset + node.depth - 1) for node in candidate_nodes], dtype=torch.long).unsqueeze(0)
    
    return tree_input_ids, tree_position_ids, tree_mask


def make_tree_attention_mask(
        prefix_len :int,
        gen_len :int,
        ancestors :list[list[int]],
        device ="cpu",
        dtype = torch.float32
    ) -> torch.FloatTensor:
    tree_mask = torch.full((gen_len, gen_len + prefix_len), torch.finfo(dtype).min, dtype=dtype).to(device=device)
    for idx, ancestor in enumerate(ancestors):
        if len(ancestor) > 0:
            tree_mask[idx][ancestor] = 0.0
    return tree_mask[None, None, :, :]


def get_residual(p: torch.Tensor, q:torch.Tensor):
    residual = (p - q).relu_()
    residual = residual / (residual.sum(dim=-1).unsqueeze(-1))
    return residual


def verify_topk(p, q, node):
    p = p.to(torch.float32)
    child_ids = torch.tensor([child.id for child in node.children], dtype=torch.long)
    for child_id in child_ids:
        r = torch.rand(1).item()
        if r <= p[child_id]:
            return True, child_id.item()
        
        else:
            p[child_id] = 0
            p = p / p.sum()
            
    return False, p.multinomial(num_samples=1).item()
   

def verify_k(p, q, node):
    p = p.to(torch.float32)
    q = q.to(torch.float32)
    child_ids = torch.tensor([child.id for child in node.children], dtype=torch.long)
    tried_ids = torch.full_like(child_ids, -1)
    for i, child_id in enumerate(child_ids):
        r = torch.rand(1).item()
        if p[child_id] > r*q[child_id]:
            return True, child_id.item()
        
        else:
            tried_ids[i] = child_id
            p = get_residual(p, q)
            
            q[child_id] = 0
            if q.sum() == 0:
                q = torch.zeros_like(q)
                q[child_ids] = 1
                q[tried_ids[tried_ids != -1]] = 0
            q = q / q.sum()
            
    return False, p.multinomial(num_samples=1).item() 
    

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