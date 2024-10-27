import torch

def verify_deterministic(p, q, node):
    child_ids = [child.id for child in node.children]
    truth_token_id = p.argmax().item()
    if truth_token_id in child_ids:
        return truth_token_id, None
    else:
        return None, p

def verify_fast(p, q, node):
    child_ids = torch.tensor([child.id for child in node.children], dtype=torch.long)
    sampled_token_id = p.multinomial(num_samples=1).item()
    if sampled_token_id in child_ids:
        return sampled_token_id, None
    p = p / (1.0 - p[child_ids].sum())
    p[child_ids] = 0
    return None, p

def verify_topk(p, q, node):
    child_ids = [child.id for child in node.children]
    for child_id in child_ids:
        r = torch.rand(1).item()
        if r <= p[child_id]:
            return child_id, None
        else:
            p = p / (1.0 - p[child_id])
            p[child_id] = 0
    return None, p

def get_residual(p: torch.Tensor, q: torch.Tensor):
    residual = (p - q).relu_()
    residual = residual / residual.sum(dim=-1, keepdim=True)
    return residual

def verify_k(p, q, node):
    child_ids = [child.id for child in node.children]
    tried_ids = torch.full((len(child_ids),), -1, dtype=torch.long)
    for i, child_id in enumerate(child_ids):
        r = torch.rand(1).item()
        if p[child_id] > r*q[child_id]:
            return child_id, None
        else:
            tried_ids[i] = child_id
            p = get_residual(p, q)
            
            q[child_id] = 0
            if q.sum() == 0:
                q.zero_()
                q[child_ids] = 1
                q[tried_ids[tried_ids != -1]] = 0
            q = q / q.sum()   
    return None, p

def verify_step(p, q, node, do_sample):
    verify_methods = {
        "deterministic": verify_deterministic,
        "fast": verify_fast,
        "greedy": verify_topk,
        "stochastic": verify_k
    }
    method_name = "deterministic" if not do_sample else node.verify_method
    _verify_step = verify_methods.get(method_name)
    if _verify_step is None:
        raise ValueError(f"Unknown verify method: {method_name}")
    return _verify_step(p, q, node)