import torch
from bigtree import Node


# def sampling_without_replacement(
#         sampling_probs: torch.Tensor,
#         rand: torch.Tensor,
#         num_samples: int
#     ):
#     # Ensure that sampling_probs contains no zeros to avoid log(0)
#     sampling_probs = torch.clamp(sampling_probs, min=1e-10)
#     # Use Gumbel-max trick for sampling without replacement
#     gumbel_noise = -torch.log(-torch.log(rand))
#     scores = torch.log(sampling_probs) + gumbel_noise
#     sampled_indices = scores.topk(k=num_samples, dim=1).indices
#     sampled_probs = torch.gather(sampling_probs, 1, sampled_indices)
#     return sampled_indices, sampled_probs

# Modified from https://github.com/Infini-AI-Lab/Sequoia/blob/main/utils.py
def sampling_without_replacement(
        sampling_probs: torch.Tensor, 
        rand: torch.Tensor,  
        num_samples: int
    ):
    sampled_indices = (rand.log()/sampling_probs).topk(k=num_samples).indices
    sampled_probs = torch.gather(sampling_probs, 1, sampled_indices)
    
    return sampled_indices, sampled_probs


def balls_to_bins(
        sampling_probs: torch.Tensor,
        bins: int,
        num_samples: int,
        do_sample: bool = True,
    ):
    if do_sample:
        sampled_bin_ids = torch.multinomial(sampling_probs, num_samples, replacement=True)
        sampled_bin_counts = torch.bincount(sampled_bin_ids, minlength=bins)
    else:
        sampled_bin_counts = torch.floor(sampling_probs * num_samples).int()
        # Adjust counts to ensure the sum equals num_samples
        count_diff = num_samples - sampled_bin_counts.sum()
        if count_diff != 0:
            residual_probs = sampling_probs - sampled_bin_counts.float() / num_samples
            residual_probs = torch.clamp(residual_probs, min=0)
            indices = torch.argsort(residual_probs, descending=True)
            for idx in indices:
                if count_diff == 0:
                    break
                adjustment = min(count_diff, 1) if count_diff > 0 else max(count_diff, -1)
                sampled_bin_counts[idx] += adjustment
                count_diff -= adjustment
    return sampled_bin_counts


def topk_sampling(sampling_probs, nodes, num_samples, step, min_accept_prob=1e-8):
    parent_probs = torch.tensor([node.global_prob for node in nodes],
                                dtype=sampling_probs.dtype,
                                device=sampling_probs.device).unsqueeze(1)
    global_probs = sampling_probs * parent_probs  # Global probabilities
    flattened_probs = global_probs.view(-1)
    topk_values, topk_indices = torch.topk(flattened_probs, num_samples)
    prev_inds = topk_indices // sampling_probs.shape[1]
    token_ids = topk_indices % sampling_probs.shape[1]

    # Create nodes
    next_nodes = []
    last_node_id = int(nodes[-1].name)
    for idx in range(num_samples):
        prev_ind = prev_inds[idx].item()
        token_id = token_ids[idx].item()
        global_prob = topk_values[idx].item()
        if global_prob < min_accept_prob:
            continue
        
        prev_node = nodes[prev_ind]
        prev_node.sample_probs = sampling_probs[prev_ind]
        prev_node.verify_method = "greedy"
        
        prob = global_prob / prev_node.global_prob
        last_node_id += 1
        new_node = Node(str(last_node_id), id=token_id, prob=prob, global_prob=global_prob, ind=prev_ind)
        next_nodes.append(new_node)

    return next_nodes


def k_sampling(sampling_probs, nodes, num_samples, step, min_accept_prob=1e-8):
    rand = torch.rand_like(sampling_probs)
    sampled_indices, sampled_probs = sampling_without_replacement(
        sampling_probs, rand=rand, num_samples=num_samples)
    # Assign tokens to parents based on normalized global probabilities
    parent_probs = torch.tensor([node.global_prob for node in nodes],
                                dtype=sampling_probs.dtype,
                                device=sampling_probs.device)
    parent_probs /= parent_probs.sum()
    parent_bin_counts = balls_to_bins(parent_probs, bins=len(nodes),
                                      num_samples=num_samples, do_sample=False)

    # Create nodes
    next_nodes = []
    last_node_id = int(nodes[-1].name)
    for prev_ind, count in enumerate(parent_bin_counts):
        if count == 0:
            continue
        
        prev_node = nodes[prev_ind]
        prev_node.sample_probs = sampling_probs[prev_ind]
        prev_node.verify_method = "stochastic"
        for i in range(count):
            token_id = sampled_indices[prev_ind][i].item()
            prob = sampled_probs[prev_ind][i].item()
            global_prob = prob * prev_node.global_prob
            if global_prob < min_accept_prob:
                continue
            
            last_node_id += 1
            new_node = Node(str(last_node_id), id=token_id, prob=prob,
                            global_prob=global_prob, ind=prev_ind)
            next_nodes.append(new_node)

    return next_nodes


def heuristic_k_sampling(sampling_probs, nodes, num_samples, step, min_accept_prob=1e-8):
    rand = torch.rand_like(sampling_probs)
    sampled_indices, sampled_probs = sampling_without_replacement(
        sampling_probs, rand=rand, num_samples=num_samples)

    # Calculate global probabilities
    parent_probs = torch.tensor([node.global_prob for node in nodes],
                                dtype=sampling_probs.dtype,
                                device=sampling_probs.device).unsqueeze(1)
    global_probs = sampling_probs * parent_probs
    flattened_probs = global_probs.view(-1)
    topk_values, topk_indices = torch.topk(flattened_probs, num_samples)
    prev_inds = topk_indices // sampling_probs.shape[1]
    parent_bin_counts = torch.bincount(prev_inds, minlength=len(nodes))

    # Create nodes
    next_nodes = []
    last_node_id = int(nodes[-1].name)
    for prev_ind, count in enumerate(parent_bin_counts):
        if count == 0:
            continue
        
        prev_node = nodes[prev_ind]
        prev_node.sample_probs = sampling_probs[prev_ind]
        prev_node.verify_method = "stochastic"
        for i in range(count):
            token_id = sampled_indices[prev_ind][i].item()
            prob = sampled_probs[prev_ind][i].item()
            global_prob = prob * prev_node.global_prob
            if global_prob < min_accept_prob:
                continue
            
            last_node_id += 1
            new_node = Node(str(last_node_id), id=token_id, prob=prob,
                            global_prob=global_prob, ind=prev_ind)
            next_nodes.append(new_node)

    return next_nodes