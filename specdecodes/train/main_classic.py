import argparse
import gc
import math
import os
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List

from transformers import LlamaForCausalLM, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.cache_utils import StaticCache, DynamicCache

from accelerate import Accelerator
from accelerate.utils import set_seed, release_memory

from tqdm import tqdm
from copy import deepcopy
import wandb

from ..models import SSM_Classic
# from liger_kernel.transformers import apply_liger_kernel_to_llama


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
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])        
        batch_loss_mask = torch.cat([self.paddingtensor2D(item['loss_mask'], max_length) for item in features])
        batch_attention_mask = torch.cat([self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        
        
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
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

@torch.no_grad()
def getkacc(model, data, lm_head, max_length=5):
    # generate future tokens
    def generate(hidden_states, input_ids, lm_head, max_length):
        past_key_values = DynamicCache()
        for i in range(max_length):
            outputs = model(
                input_ids=input_ids if i == 0 else token,
                past_key_values=past_key_values, 
                use_cache=True
            )
            hidden_states = outputs.last_hidden_state[:, -1:].clone()
            del outputs
            
            token = torch.argmax(lm_head(hidden_states), dim=-1)
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
    target_ids = lm_head(target).argmax(dim=2)
    
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
        out_ids = generate(pre_hidden_states, pre_input_ids, lm_head, max_length=max_length)
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
def update_metrics(loss_mask, s_out, t_out, correct, total, topk_acc):
    _, predicted = torch.max(s_out, 2)
    _, targeted = torch.max(t_out, 2)
    correct += ((predicted == targeted) * loss_mask).sum().item()
    total += loss_mask.sum().item()
    
    # Calculate expectation
    p_t = F.softmax(t_out, dim=-1)
    p_s = F.softmax(s_out, dim=-1)
    expect = torch.sum(p_t * p_s, dim=-1).mean()
    
    # Calculate top-k accuracy
    s_out = s_out.view(-1, t_out.shape[-1])[loss_mask.view(-1)]
    targeted = targeted.view(-1)[loss_mask.view(-1)]
    temp_top_acc = top_accuracy(s_out, targeted, (1, 2, 3))
    for idx, top_i in enumerate(temp_top_acc):
        topk_acc[idx] += top_i

    return expect, correct, total

def aggr_mean_loss(mask, loss):
    return torch.sum(torch.mean(mask.unsqueeze(-1) * loss, 2)) / (mask.sum() + 1e-5)

def aggr_sum_loss(mask, loss):
    return torch.sum(torch.sum(mask.unsqueeze(-1) * loss, 2)) / (mask.sum() + 1e-5)

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

def calculate_loss(loss_mask, s_logits, t_logits, s_out, t_out, train_config):
    ploss = calc_kl_loss(loss_mask, s_out, t_out)
    # vloss = calc_l1_loss(loss_mask, s_logits, t_logits)
    # vloss = calc_skl_loss(loss_mask, s_logits, t_logits)
    loss = ploss#train_config["p_w"] * ploss + train_config["v_w"] * vloss
    
    return loss, ploss, ploss#vloss

def train_one_epoch(model, lm_head, train_loader, optimizer, scheduler, train_config, epoch, num_epochs, accelerator, run=None):
    model.train()
    device = accelerator.device
    correct, total, epoch_loss, num_batches = 0, 0, 0, 0
    topk_acc = [0] * 3

    for idx, data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            with torch.no_grad():
                t_logits = data["target"] # data["target"] = data["hidden_states"][:, 1:]
                t_out = lm_head(t_logits.to(lm_head.weight.dtype))
            s_logits = model(input_ids=data["input_ids"], attention_mask=data["attention_mask"])[0]
            s_out = lm_head(s_logits)
            
            loss, ploss, vloss = calculate_loss(data["loss_mask"], s_logits, t_logits, s_out, t_out, train_config)
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            scheduler.step()

        prev_total = total
        expect, correct, total = update_metrics(data["loss_mask"], s_out, t_out, correct, total, topk_acc)
        if accelerator.is_main_process and (idx % train_config["log_freq"] == 0) and total > prev_total and run:
            logdict = {
                "train/lr": optimizer.optimizer.param_groups[0]["lr"], 
                "train/vloss": vloss.item(),
                "train/ploss": ploss.item(), 
                "train/loss": loss.item(), 
                "train/acc": correct / total,
                "train/expect": expect.item()
            }
            for id, acc in enumerate(topk_acc):
                logdict[f'train/top_{id + 1}_acc'] = acc.item() / total
            run.log(logdict)
        epoch_loss += loss.item()
        num_batches += 1

    # gather metrics
    correct, total = torch.tensor(correct, device=device), torch.tensor(total, device=device)
    correct, total = accelerator.gather_for_metrics((correct, total))
    expect = accelerator.gather_for_metrics(expect)
    topk_acc = accelerator.gather_for_metrics(topk_acc)
    
    expect = expect.mean().item()
    correct = correct.sum().item()
    total = total.sum().item()
    epoch_loss /= num_batches

    if accelerator.is_local_main_process and run:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        print(f'Train Accuracy: {100 * correct / total:.2f}%')
        logdict = {
            "train/epochacc": correct / total, 
            "train/epochloss": epoch_loss,
            "train/epochexpect": expect,
        }
        for id, acc in enumerate(topk_acc):
            logdict[f'train/epochtop_{id + 1}_acc'] = acc.sum().item() / total
        run.log(logdict)


@torch.no_grad()
def validate(model, lm_head, test_loader, train_config, epoch, num_epochs, save_dir, accelerator, run=None):
    device = accelerator.device
    model.eval()
    correct, total, epoch_loss, num_batches = 0, 0, 0, 0
    topk_acc = [0] * 3

    for batch_idx, data in enumerate(tqdm(test_loader, desc="Validating")):
        # Forward pass
        s_logit = model(input_ids=data["input_ids"], attention_mask=data["attention_mask"])[0]
        t_logit = data["target"]

        # Head computation
        t_out = lm_head(t_logit.to(lm_head.weight.dtype))
        s_out = lm_head(s_logit)

        # Calculate loss
        loss, ploss, vloss = calculate_loss(data["loss_mask"], s_logit, t_logit, s_out, t_out, train_config)

        # Update metrics
        expect, correct, total = update_metrics(data["loss_mask"], s_out, t_out, correct, total, topk_acc)
        epoch_loss += loss.item()
        num_batches += 1

    # gather metrics
    correct, total = torch.tensor(correct, device=device), torch.tensor(total, device=device)
    correct, total = accelerator.gather_for_metrics((correct, total))
    expect = accelerator.gather_for_metrics(expect)
    topk_acc = accelerator.gather_for_metrics(topk_acc)
    
    expect = expect.mean().item()
    correct = correct.sum().item()
    total = total.sum().item()
    epoch_loss /= num_batches

    # Log and save model
    if accelerator.is_local_main_process and run:
        print(f'Test Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        print(f'Test Accuracy: {100 * correct / total:.2f}%')
        logdict = {
            "test/epochacc": correct / total, 
            "test/epochloss": epoch_loss,
            "test/ploss": ploss.item(),
            "test/vloss": vloss.item(),
            "test/expect": expect,
        }

        for id, acc in enumerate(topk_acc):
            logdict[f'test/top_{id + 1}_acc'] = acc.sum().item() / total
        run.log(logdict)

        # save model
        accelerator.save_model(model, f"{save_dir}/model_{epoch + 1}")
        

def main(args):
    set_seed(0) # fix seed

    # HUGE speedup, especially on A100 or above
    torch.backends.cuda.matmul.allow_tf32 = True

    train_config = {
        "p_w": 0.1,
        "v_w": 1.0,
        "num_workers": 4,
        "data_noise": True,
        # "data_noise": False,
        "noise": "uniform",
        "mean": 0.0,
        "std": 0.2,
        # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
        "max_len": 2048,
        "grad_clip": 1.0,
        "save_freq": 5,
        "log_freq": 1,
    }
    # merge train_config with args
    train_config.update(vars(args))

    # init Accelerator
    accelerator = Accelerator()
    
    # wandb
    run = None
    if accelerator.is_main_process:
        if not args.wandb:
            os.environ['WANDB_DISABLED'] = 'true'

        # Add requirement for wandb core
        wandb.require("core")
        run = wandb.init(project="eagle", config=train_config)

    # NEFTune
    if train_config["data_noise"]:
        if train_config["noise"] == "uniform":
            aug = AddUniformNoise(std=train_config["std"])
        else:
            aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
    else:
        aug = None

    # Load dataset
    datapath = list_files(args.datadir)
    datapath = datapath[:int(len(datapath) * args.data_ratio)]
    print('Total data:',len(datapath))

    traindatapath = datapath[:int(len(datapath) * 0.95)]
    testdatapath = datapath[int(len(datapath) * 0.95):]

    traindataset = CustomDataset(traindatapath, transform=aug, max_len=train_config["max_len"])
    testdataset = CustomDataset(testdatapath, max_len=train_config["max_len"])

    train_loader = DataLoader(traindataset, batch_size=args.bs, shuffle=True,
                            collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                            pin_memory=True, drop_last=True)
    test_loader = DataLoader(testdataset, batch_size=args.bs, shuffle=False,
                            collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)
    
    # create folder for saving model
    if accelerator.is_main_process:
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)


    # load head
    print("Loading head...")
    config = AutoConfig.from_pretrained(args.llm_path)
    llm = LlamaForCausalLM.from_pretrained(
        config=config,
        pretrained_model_name_or_path=args.llm_path,
        torch_dtype=torch.float32,
        # low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    draft_config = deepcopy(llm.config)
    draft_config.num_hidden_layers = args.layers
    draft_config.use_cache = False
    draft_config._attn_implementation = "sdpa"
    
    # create new head for draft model
    lm_head = nn.Linear(draft_config.hidden_size, draft_config.vocab_size, bias=False)
   
    # load weights from llm
    lm_head.weight.data = llm.lm_head.weight.data
    
    # not traininable
    for param in lm_head.parameters():
        param.requires_grad = False
    
    # convert to fp32
    lm_head = lm_head.to(torch.float32)
    
    # load weights from pretrained model if specified
    if args.pretrained is not None:
        print("Loading pretrained model...")
        model = SSM_Classic.from_pretrained(args.pretrained, config=draft_config)
    else:
        print("Loading draft model...")
        model = SSM_Classic(config=draft_config)
    
    # load llm's embeddings to draft model
    for param, llm_param in zip(model.model.embed_tokens.parameters(), llm.get_input_embeddings().parameters()):
        param.data = llm_param.data
        param.requires_grad = False
    
    # apply liger kernel to ssm model (however causes error during training, not sure why)
    # apply_liger_kernel_to_llama(model=model.model)
    
    # load llm's last attention layer's data to draft model (not trainable)
    # load_index = -1
    # for (draft_param, llm_param) in zip(model.model.layers[load_index].parameters(), llm.model.layers[load_index].parameters()):
    #     draft_param.data = llm_param.data
    #     draft_param.requires_grad = False
    
    # # load llm's first layer's data to draft model
    # load_index = 0
    # for (draft_param, llm_param) in zip(model.model.layers[load_index].parameters(), llm.model.layers[load_index].parameters()):
    #     draft_param.data = llm_param.data
    #     draft_param.requires_grad = False
        
    # Manually free up memory before loading pretrained model
    print(f'Current used GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3} GB')
    del llm
    release_memory()
    print(f'GPU memory after freeing llm: {torch.cuda.memory_allocated() / 1024 ** 3} GB')

    # Calculate the number of update steps per epoch  https://github.com/huggingface/diffusers/pull/6143/files
    print("Setting up training...")
    num_update_steps_per_epoch = math.ceil(len(train_loader) / accelerator.num_processes / accelerator.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    num_warmup_steps = max_train_steps * args.warmup_ratio * accelerator.num_processes
    num_training_steps = max_train_steps * accelerator.num_processes
    print(f'warmup steps: {num_warmup_steps}, training steps: {num_training_steps}')
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    # Prepare accelerator
    print("Preparing accelerator...")
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )
    
    # move to device
    lm_head = lm_head.to(accelerator.device)
        
 
    # Training loop
    print("Start training...")
    for epoch in range(args.epochs):
        # Train
        train_one_epoch(
            model, lm_head,
            train_loader, optimizer, scheduler, train_config, 
            epoch, args.epochs, accelerator, run
        )
        
        # Validate
        if not args.no_validate:
            if (epoch == args.epochs-1) or (epoch % train_config["save_freq"] == 0):
                validate(
                    model, lm_head,
                    test_loader, train_config, 
                    epoch, args.epochs, args.savedir, accelerator, run
                )

    # Finish
    if accelerator.is_main_process:
        run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sp')
    parser.add_argument('--llm-path', '-llm', type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument('--datadir', type=str, default='0')
    parser.add_argument('--savedir', type=str, default='0')
    parser.add_argument('--data-ratio', type=float, default=1)
    parser.add_argument('--pretrained', type=str, default=None)
    
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--warmup-ratio', type=int, default=0.05)
    parser.add_argument('--bs', type=int, default=4)
    # https://github.com/NVIDIA/NeMo/blob/876c8511e579c1c343b52bdd96ebe2296608434c/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb
    parser.add_argument('--lr', type=float, default=1e-4) #1e-2 too large, loss becomes nan when lr>3e-4
    parser.add_argument('--weight-decay', type=float, default=1e-2) # from paper: 1e-3
    parser.add_argument('--betas', type=float, default=(0.9, 0.95))  # from paper: (0.95, 0.5)
    
    parser.add_argument('--no-validate', '-nv', action='store_true', help='Skip validation')
    
    # logging
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    main(args)

    