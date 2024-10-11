import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from typing import Any, Dict, List
from transformers.cache_utils import StaticCache, DynamicCache

import math
import os

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data

class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data

class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None, max_len=-1):
        self.data = datapath
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index], weights_only=False)
        new_data = {}
        
        hidden_state = data['hidden_state'][:self.max_len].unsqueeze(0)
        input_ids = data['input_ids'][:self.max_len].unsqueeze(0)
        loss_mask = data["loss_mask"][:self.max_len].to(dtype=torch.bool).unsqueeze(0)

        length = hidden_state.shape[1]
        attention_mask = torch.ones(1, length, dtype=torch.bool)
        
        # mask the last token
        loss_mask[0, -1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.zeros(1, 1, dtype=input_ids.dtype)
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target
        
        if self.transform:
            new_data = self.transform(new_data)

        return new_data

class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_prev_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])        
        batch_loss_mask = torch.cat([self.paddingtensor2D(item['loss_mask'], max_length) for item in features])
        batch_attention_mask = torch.cat([self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_prev_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch

def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

# Currently not used
@torch.no_grad()
def getkacc(model, data, llm_last, max_length=5):
    # generate future tokens
    def generate(hidden_states, input_ids, llm_last, max_length):
        past_key_values = DynamicCache()
        for i in range(max_length):
            outputs = model(
                input_ids=input_ids if i == 0 else token,
                past_key_values=past_key_values, 
                use_cache=True
            )
            hidden_states = outputs.last_hidden_state[:, -1:].clone()
            del outputs
            
            token = torch.argmax(llm_last(hidden_states), dim=-1)
            input_ids = torch.cat((input_ids, token), dim=-1)
        return input_ids
    
    # get data
    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    target = data["target"]
    
    # get batch size and sequence length
    bs, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    
    # generate target ids
    target_ids = llm_last(target).argmax(dim=2)
    
    # for calculating accuracy
    total = torch.zeros(max_length, dtype=torch.int32)
    correct = torch.zeros(max_length, dtype=torch.int32)

    # iterate through each prefix length
    for pre_len in range(1, seq_len):
        if loss_mask[:, pre_len].sum().item() == 0:
            continue
        
        # generate future tokens
        pre_hidden_states = hidden_states[:, :pre_len]
        pre_input_ids = input_ids[:, :pre_len]
        out_ids = generate(pre_hidden_states, pre_input_ids, llm_last, max_length=max_length)
        generate_ids = out_ids[:, pre_len:]
        
        # calculate accuracy
        for bid in range(bs):
            for k in range(max_length):
                if (loss_mask[bid, pre_len + k] == 0) or (pre_len + k >= seq_len):
                    break
                
                total[k] += 1
                if generate_ids[bid, k] == target_ids[bid, pre_len + k - 1]:
                    correct[k] += 1
                else:
                    total[k+1:max_length] += 1
                    break

    acc = correct.float() / total.float()
    return acc.cpu().tolist()

@torch.no_grad()
def update_metrics(loss_mask, s_logits, t_logits, correct, total, topk_acc):
    _, predicted = torch.max(s_logits, 2)
    _, targeted = torch.max(t_logits, 2)
    correct += ((predicted == targeted) * loss_mask).sum().item()
    total += loss_mask.sum().item()
    
    # Calculate expectation
    p_t = F.softmax(t_logits, dim=-1)
    p_s = F.softmax(s_logits, dim=-1)
    expect = torch.sum(p_t * p_s, dim=-1).mean()
    
    # Calculate top-k accuracy
    s_logits = s_logits.view(-1, t_logits.shape[-1])[loss_mask.view(-1)]
    targeted = targeted.view(-1)[loss_mask.view(-1)]
    temp_top_acc = top_accuracy(s_logits, targeted, (1, 2, 3))
    for idx, top_i in enumerate(temp_top_acc):
        topk_acc[idx] += top_i

    return expect, correct, total

def aggr_mean_loss(mask, loss):
    return torch.sum(torch.mean(mask.unsqueeze(-1) * loss, 2)) / (mask.sum() + 1e-5)

def aggr_sum_loss(mask, loss):
    return torch.sum(torch.sum(mask.unsqueeze(-1) * loss, 2)) / (mask.sum() + 1e-5)

def calc_ce_loss(mask, s_logits, t_logits):
    with torch.no_grad():
        t_logits = F.softmax(t_logits, dim=-1)
        
    s_logits = F.log_softmax(s_logits, dim=-1)
    loss = -t_logits * s_logits
    loss = aggr_sum_loss(mask, loss)
    return loss

def calc_kl_loss(mask, s_logits, t_logits):
    with torch.no_grad():
        t_logits = F.softmax(t_logits, dim=-1)
    s_logits = F.log_softmax(s_logits, dim=-1)
    loss = F.kl_div(s_logits, t_logits, reduction='none')
    loss = aggr_sum_loss(mask, loss)
    return loss

def calc_kl_loss_with_temp(mask, s_logits, t_logits, temp=2.5):
    with torch.no_grad():
        t_logits = F.softmax(t_logits/temp, dim=-1)
    s_logits = F.log_softmax(s_logits/temp, dim=-1)
    loss = F.kl_div(s_logits, t_logits, reduction='none')
    loss = aggr_sum_loss(mask, loss)
    return loss * (temp ** 2)

def calc_skl_loss(mask, s_logits, t_logits, alpha=0.1):
    with torch.no_grad():
        t_logits = F.softmax(t_logits, dim=-1)
    t_logits = alpha * t_logits + (1 - alpha) * F.softmax(s_logits, dim=-1)
    s_logits = F.log_softmax(s_logits, dim=-1)
    
    loss = F.kl_div(s_logits, t_logits, reduction='none')
    loss = aggr_sum_loss(mask, loss)
    return loss

def calc_rkl_loss(mask, s_logits, t_logits):
    with torch.no_grad():
        t_logits = F.log_softmax(t_logits, dim=-1)
    s_logits = F.softmax(s_logits, dim=-1)
    
    loss = F.kl_div(t_logits, s_logits, reduction='none')
    loss = aggr_sum_loss(mask, loss)
    return loss

def calc_tvdpp_loss(mask, s_logits, t_logits):
    s_logits = F.softmax(s_logits, dim=-1)
    with torch.no_grad():
        t_logits = F.softmax(t_logits, dim=-1)
        reward = torch.as_tensor((t_logits-s_logits) > 0, dtype=torch.float32)
        std = torch.std(reward, dim=-1, keepdim=True)
        mean = torch.mean(reward, dim=-1, keepdim=True)
        reward = (reward - mean) / (std + 1e-5)
    
    policy_loss = torch.log(s_logits) * -reward
    policy_loss = aggr_mean_loss(mask, policy_loss)
    
    return policy_loss

def calc_tv_loss(mask, s_logits, t_logits):
    with torch.no_grad():
        t_logits = F.softmax(t_logits, dim=-1)
    s_logits = F.softmax(s_logits, dim=-1)
    
    loss = F.l1_loss(s_logits, t_logits, reduction='none')
    loss = aggr_sum_loss(mask, loss) * 0.5
    return loss

def calc_tv_loss_with_temp(mask, s_logits, t_logits, temp=2.5):
    with torch.no_grad():
        t_logits = F.softmax(t_logits/temp, dim=-1)
    s_logits = F.softmax(s_logits/temp, dim=-1)
    
    loss = F.l1_loss(s_logits, t_logits, reduction='none')
    loss = aggr_sum_loss(mask, loss) * 0.5
    return loss * (temp ** 2)

def calc_l1_loss(mask, s_logits, t_logits):
    loss = F.l1_loss(s_logits, t_logits, reduction='none')
    loss = aggr_mean_loss(mask, loss)
    return loss

def calc_smooth_l1_loss(mask, s_logits, t_logits):
    loss = F.smooth_l1_loss(s_logits, t_logits, reduction='none')
    loss = aggr_mean_loss(mask, loss)
    return loss