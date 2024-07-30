
import torch

from bigtree import levelorder_iter

def invert_mask(mask, dtype=torch.float32):
    inverted_mask = ~mask
    mask = inverted_mask.to(dtype=dtype).masked_fill(
        inverted_mask, torch.finfo(dtype).min
    )
    return mask

def build_tree_attention_data(root, position_offset):
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
    tree_mask = invert_mask(tree_mask)

    # tree_candidates = node id
    tree_input_ids = torch.tensor([node.id for node in candidate_nodes])[None]
    
    # tree_position_id = node depth + offset
    tree_position_ids = torch.tensor([(position_offset + node.depth - 1) for node in candidate_nodes], dtype=torch.long)[None]
    
    return tree_input_ids, tree_position_ids, tree_mask