import torch
from ..utils import get_residual

def verify_deterministic(p, q, node):
    child_ids = [child.id for child in node.children]
    truth_token_id = p.argmax().item()
    if truth_token_id in child_ids:
        return truth_token_id, None
    
    else:
        return None, p

def verify_fast(p, q, node):
    p = p.to(torch.float32)
    child_ids = [child.id for child in node.children]
    truth_token_id = p.multinomial(num_samples=1).item()
    if truth_token_id in child_ids:
        return truth_token_id, None
    
    else:
        for child_id in child_ids:
            p[child_id] = 0
            p = p / p.sum()
            
        return None, p

def verify_topk(p, q, node):
    p = p.to(torch.float32)
    child_ids = [child.id for child in node.children]
    for child_id in child_ids:
        r = torch.rand(1).item()
        if r <= p[child_id]:
            return child_id, None
        
        else:
            p[child_id] = 0
            p = p / p.sum()
            
    return None, p

def verify_k(p, q, node):
    p = p.to(torch.float32)
    q = q.to(torch.float32)
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
                q = torch.zeros_like(q)
                q[child_ids] = 1
                q[tried_ids[tried_ids != -1]] = 0
            q = q / q.sum()
            
    return None, p