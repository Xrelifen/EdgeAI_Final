import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List
import os

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_states"]
        noise = torch.randn_like(tensor) * self.std + self.mean
        data["hidden_states"] = tensor + noise
        return data

class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_states"]
        scale = self.std * 512 / tensor.shape[0]
        noise = (torch.rand_like(tensor) - 0.5) * scale
        data["hidden_states"] = tensor + noise
        return data

class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None, max_len=-1):
        self.data = datapath
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index], weights_only=False)
        
        # rename 'hidden_state' to 'hidden_states' after rebuilding the dataset
        hidden_states = data['hidden_state'][:self.max_len]
        input_ids = data['input_ids'][:self.max_len]
        loss_mask = data["loss_mask"][:self.max_len].to(dtype=torch.bool)

        length = hidden_states.shape[0]
        attention_mask = torch.ones(length, dtype=torch.bool, device=hidden_states.device)
        
        # Mask the last token
        if length > 0:
            loss_mask[-1] = 0

        input_ids_target = torch.cat(
            (input_ids[1:], torch.zeros(1, dtype=input_ids.dtype, device=input_ids.device)), dim=0
        )
        target = torch.cat(
            (hidden_states[1:], torch.zeros(1, hidden_states.shape[1], device=hidden_states.device)), dim=0
        )
        
        new_data = {
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "target": target,
            "hidden_states": hidden_states,
            "input_ids": input_ids_target
        }
        
        if self.transform:
            new_data = self.transform(new_data)

        return new_data

class DataCollatorWithPadding:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        keys = features[0].keys()
        for key in keys:
            tensors = [f[key] for f in features]
            if key in ['hidden_states', 'target']:
                batch[key] = pad_sequence(tensors, batch_first=True)
            elif key == 'input_ids':
                batch[key] = pad_sequence(tensors, batch_first=True, padding_value=self.pad_token_id)
            elif key in ['attention_mask', 'loss_mask']:
                batch[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
            else:
                batch[key] = pad_sequence(tensors, batch_first=True)
        return batch

def list_files(path):
    return [os.path.join(root, file)
            for root, _, files in os.walk(path)
            for file in files]

def top_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # Shape: (batch_size, maxk)
        pred = pred.t()  # Shape: (maxk, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k)
        return res

def top_sampled_probability_sum(output, target, topk=(1,)):
    """Computes the sum of target probabilities over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        _, topk_indices = output.topk(maxk, dim=1, largest=True, sorted=True)
        res = []
        for k in topk:
            topk_probs = torch.gather(target, 1, topk_indices[:, :k])
            prob_sum = topk_probs.sum()
            res.append(prob_sum)
        return res

@torch.no_grad()
def update_metrics(loss_mask, s_logits, t_logits, topk: List[int] = [1, 3, 5, 9]):
    predicted = torch.argmax(s_logits, dim=2)
    targeted = torch.argmax(t_logits, dim=2)
    correct = ((predicted == targeted) * loss_mask).sum().item()
    total = loss_mask.sum().item()
    
    # Calculate expectation
    p_s = F.softmax(s_logits, dim=-1)
    p_t = F.softmax(t_logits, dim=-1)
    expect = (torch.sum(p_s * p_t, dim=-1) * loss_mask).sum().item()
    
    # Flatten tensors where loss_mask is True
    mask_flat = loss_mask.view(-1)
    p_s_flat = p_s.view(-1, p_s.shape[-1])[mask_flat]
    p_t_flat = p_t.view(-1, p_t.shape[-1])[mask_flat]
    
    topk_probs = {}
    temp_top_probs_sum = top_sampled_probability_sum(p_s_flat, p_t_flat, topk)
    for idx, k in enumerate(topk):
        topk_probs[f"prob@{k}"] = temp_top_probs_sum[idx].item()

    return correct, total, expect, topk_probs
    
def aggr_mean_loss(mask, loss):
    return torch.sum(torch.mean(loss * mask.unsqueeze(-1), 2)) / (mask.sum() + 1e-5)

def aggr_sum_loss(mask, loss):
    return torch.sum(torch.sum(loss * mask.unsqueeze(-1), 2)) / (mask.sum() + 1e-5)

def calc_ce_loss(mask, s_logits, t_logits):
    with torch.no_grad():
        t_probs = F.softmax(t_logits, dim=-1)
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    loss = -t_probs * s_log_probs
    return aggr_sum_loss(mask, loss)

def calc_kl_loss(mask, s_logits, t_logits):
    with torch.no_grad():
        t_probs = F.softmax(t_logits, dim=-1)
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    loss = F.kl_div(s_log_probs, t_probs, reduction='none')
    return aggr_sum_loss(mask, loss)

def calc_kl_loss_with_temp(mask, s_logits, t_logits, temp=2.5):
    with torch.no_grad():
        t_probs = F.softmax(t_logits / temp, dim=-1)
    s_log_probs = F.log_softmax(s_logits / temp, dim=-1)
    loss = F.kl_div(s_log_probs, t_probs, reduction='none')
    return aggr_sum_loss(mask, loss) * (temp ** 2)

def calc_skl_loss(mask, s_logits, t_logits, alpha=0.1):
    with torch.no_grad():
        t_probs = F.softmax(t_logits, dim=-1)
    s_probs = F.softmax(s_logits, dim=-1)
    mixed_probs = alpha * t_probs + (1 - alpha) * s_probs
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    loss = F.kl_div(s_log_probs, mixed_probs, reduction='none')
    return aggr_sum_loss(mask, loss)

def calc_rkl_loss(mask, s_logits, t_logits):
    with torch.no_grad():
        t_log_probs = F.log_softmax(t_logits, dim=-1)
    s_probs = F.softmax(s_logits, dim=-1)
    loss = F.kl_div(t_log_probs, s_probs, reduction='none')
    return aggr_sum_loss(mask, loss)

def calc_l1_loss(mask, s_logits, t_logits):
    loss = F.l1_loss(s_logits, t_logits, reduction='none')
    return aggr_mean_loss(mask, loss)

def calc_sl1_loss(mask, s_logits, t_logits):
    loss = F.smooth_l1_loss(s_logits, t_logits, reduction='none')
    return aggr_mean_loss(mask, loss)
