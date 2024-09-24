import torch
from bigtree import Node


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
        # handle the case where the sum of the counts is more or less than num_samples
        count_sum = sampled_bin_counts.sum()
        if count_sum < num_samples:
            _, top_indices = torch.topk(sampling_probs, num_samples - count_sum)
            sampled_bin_counts[top_indices] += 1
    return sampled_bin_counts


def topk_sampling(sampling_probs, nodes, num_samples, step):
    n, vocab_dim = sampling_probs.shape
    parent_probs = torch.tensor([node.prob for node in nodes], dtype=sampling_probs.dtype, device=sampling_probs.device).unsqueeze(1)
    global_probs = sampling_probs * parent_probs # Multiply by parent's prob to get global prob
    flattened_probs = global_probs.flatten()
    # Get the indices of the top k values
    topk_values, topk_indices = torch.topk(flattened_probs, num_samples)
    prev_indices = topk_indices // vocab_dim
    token_ids = topk_indices % vocab_dim
    
    #* Create nodes
    next_nodes = []
    last_node_id = int(nodes[-1].name)
    for prev_ind, token_id, global_prob in zip(prev_indices, token_ids, topk_values):
        prev_node = nodes[prev_ind]
        prev_node.sample_probs = sampling_probs[prev_ind]
        prob = global_prob / prev_node.prob
        # if prob < 1e-4 or global_prob < 1e-6:
            #     continue
        last_node_id += 1
        new_node = Node(str(last_node_id), id=token_id.item(), prob=prob, global_prob=global_prob, ind=prev_ind)
        next_nodes.append(new_node)

    return next_nodes


def k_sampling(sampling_probs, nodes, num_samples, step):
    rand = torch.rand(sampling_probs.shape, device=sampling_probs.device)
    sampled_indices, sampled_probs = sampling_without_replacement(sampling_probs, rand=rand, num_samples=num_samples)
    # Finding the top k tokens by parents' global prob., assign how much tokens each parent node should sample
    parent_probs = torch.tensor([node.global_prob for node in nodes], dtype=torch.float16)
    parent_probs = parent_probs / parent_probs.sum()
    parent_bin_counts = balls_to_bins(parent_probs, bins=len(nodes), num_samples=num_samples, do_sample=False)
    
    #* Create nodes
    next_nodes = []
    last_node_id = int(nodes[-1].name)
    for prev_ind, prev_node in enumerate(nodes):
        prev_node.sample_probs = sampling_probs[prev_ind]
        for i in range(parent_bin_counts[prev_ind]):
            token_id = sampled_indices[prev_ind][i]
            prob = sampled_probs[prev_ind][i]
            global_prob = prob * prev_node.prob
            # if prob < 1e-4 or global_prob < 1e-6:
            #     continue
            last_node_id += 1
            new_node = Node(str(last_node_id), id=token_id.item(), prob=prob, global_prob=global_prob, ind=prev_ind)
            next_nodes.append(new_node) 
    
    return next_nodes


# not used, effect seems to be worse than k_sampling
def test_k_sampling(sampling_probs, nodes, num_samples, step):
    rand = torch.rand(sampling_probs.shape, device=sampling_probs.device)
    sampled_indices, sampled_probs = sampling_without_replacement(sampling_probs, rand=rand, num_samples=num_samples)

    # Finding the top k tokens by childrens' global prob., assign how much tokens each parent node should sample
    n, vocab_dim = sampling_probs.shape
    parent_probs = torch.tensor([node.prob for node in nodes], dtype=sampling_probs.dtype, device=sampling_probs.device).unsqueeze(1)
    global_probs = sampling_probs * parent_probs
    flattened_probs = global_probs.flatten()
    # Get the indices of the top k values
    _, topk_indices = torch.topk(flattened_probs, num_samples)
    topk_bin_ids = topk_indices // vocab_dim  # Dividing by vocab_dim gives the bin index
    parent_bin_counts = torch.bincount(topk_bin_ids, minlength=n)
    
    #* Create nodes
    next_nodes = []
    last_node_id = int(nodes[-1].name)
    for prev_ind, prev_node in enumerate(nodes):
        prev_node.sample_probs = sampling_probs[prev_ind]
        for i in range(parent_bin_counts[prev_ind]):
            token_id = sampled_indices[prev_ind][i]
            prob = sampled_probs[prev_ind][i]
            global_prob = prob * prev_node.prob
            # if prob < 1e-4 or global_prob < 1e-6:
            #     continue
            last_node_id += 1
            new_node = Node(str(last_node_id), id=token_id.item(), prob=prob, global_prob=global_prob, ind=prev_ind)
            next_nodes.append(new_node) 
    
    return next_nodes
  

#! Currently not maintained
def cuda_graph_for_sampling_without_replacement(
    device="cuda:0", dtype=torch.float16, 
    dim=32000,
    n_warmups=3, mempool=None,
    idx_len = 8, num_samples = 16,
    temperature = 0.6,
):
    static_sampling_logits = torch.full((idx_len, dim), 1, dtype=dtype, device=device)
    static_rand = torch.empty((idx_len, dim), dtype=dtype, device=device).uniform_()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_probs, static_indices = sampling_without_replacement(
                 static_sampling_logits,
                 static_rand,
                 num_samples,
                 temperature
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_probs, static_indices = sampling_without_replacement(
                                            static_sampling_logits,
                                            static_rand,
                                            num_samples,
                                            temperature
                                        )
    
    def run(draft_logits, rand_vector):
        # static_sampling_logits.copy_(draft_logits)
        # static_rand.copy_(rand_vector)
        # graph.replay()
        # return static_position.clone()
        
        static_sampling_logits.copy_(draft_logits)
        static_rand.copy_(rand_vector)
        graph.replay()
        return static_probs.clone(), static_indices.clone()
    
    return run