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

from accelerate import Accelerator
from accelerate.utils import set_seed

from tqdm import tqdm
from copy import deepcopy
import wandb

from ..models.ssm.eagle import DraftModel

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
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['hidden_state'][:self.max_len][None, :]
        input_ids = data['input_ids'][:self.max_len][None, :]
        loss_mask = data["loss_mask"][:self.max_len][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target
        # sample = torch.cat((data['xs'],data['xb']))
        # sample=torch.cat((self.data[index]['x'],self.data[index]['logits']))
        # label = data['y']
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
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
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
def getkacc(model, data, lm_head, embed_tokens, max_length=5):
    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    target = data["target"]
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, sl = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = lm_head(target)
    hidden_states_headout = lm_head(hidden_states)

    for i in range(bs):
        for j in range(sl):

            single_hidden_states = hidden_states[i, :j]
            single_input_ids = input_ids[i, :j]

            single_hidden_states = single_hidden_states[None, :, :]
            single_input_ids = single_input_ids[None, :]
            for k in range(max_length):
                if loss_mask[i, single_hidden_states.shape[1] - 1] == 0:
                    break
                tmp_in_target_headout = hidden_states_headout[i, single_hidden_states.shape[1] - 1]
                tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1] - 1]
                target_in_token = torch.argmax(tmp_in_target_headout)
                target_out_token = torch.argmax(tmp_out_target_headout)
                tmp_token = input_ids[i, single_hidden_states.shape[1] - 1]
                # tmp_sample_mask=sample_mask[i,single_hidden_states.shape[1]-1]
                if not (target_in_token == tmp_token):
                    break
                out_hidden = model(single_hidden_states, input_ids=single_input_ids, embed_tokens=embed_tokens)[0]
                last_hidden = out_hidden[:, -1]
                last_headout = lm_head(last_hidden)
                token = torch.argmax(last_headout)
                total[k] += 1
                if token == target_out_token:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

                single_hidden_states = torch.cat((single_hidden_states, out_hidden[:, -1:]), dim=1)
                single_input_ids = torch.cat((single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)),
                                             dim=1)

    acc = [correct[i] / total[i] for i in range(len(correct))]
    return acc

def calculate_loss(predict, target, predict_head, target_head, loss_mask, criterion, train_config):
    with torch.no_grad():
        target_head_p = F.softmax(target_head, dim=2).detach()
    out_head_logp = F.log_softmax(predict_head, dim=2)
    plogp = target_head_p * out_head_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum()+1e-5)
    # ploss = F.kl_div(out_head_logp, target_head_p, reduction="none")
    # ploss = torch.sum(torch.sum(loss_mask * ploss, dim=2)) / (loss_mask.sum()+1e-5)
    
    vloss = criterion(predict, target)
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum()+1e-5)
    loss = train_config["p_w"] * ploss + train_config["v_w"] * vloss
    return loss, ploss, vloss

@torch.no_grad()
def update_metrics(predict_head, target_head, loss_mask, correct, topk_acc, total):
    _, predicted = torch.max(predict_head, 2)
    _, targeted = torch.max(target_head, 2)
    ct = loss_mask.sum().item()
    cc = ((predicted == targeted) * loss_mask.squeeze()).sum().item()
    
    predict_head = predict_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
    targeted = targeted.view(-1)[loss_mask.view(-1) == 1]

    temp_top_acc = top_accuracy(predict_head, targeted, (1, 2, 3))
    for idx, top_i in enumerate(temp_top_acc):
        topk_acc[idx] += top_i

    total += ct
    correct += cc
    return correct, total

@torch.no_grad()
def gather_metrics(correct, total, topk_acc, accelerator):
    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    topk_acc = accelerator.gather_for_metrics(topk_acc)
    return correct, total, topk_acc

def train_one_epoch(model, lm_head, embed_tokens, train_loader, optimizer, scheduler, criterion, train_config, epoch, num_epochs, accelerator, run=None):
    model.train()
    correct, total, epoch_loss, num_batches = 0, 0, 0, 0
    topk_acc = [0] * 3

    for idx, data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            predict = model(data["hidden_states"], input_ids=data["input_ids"], embed_tokens=embed_tokens, attention_mask=data["attention_mask"])[0]
            target = data["target"]

            target_head = lm_head(target.to(lm_head.weight.dtype))
            predict_head = lm_head(predict)

            loss, ploss, vloss = calculate_loss(predict, target, predict_head, target_head, data["loss_mask"][:, :, None], criterion, train_config)
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            scheduler.step()

        prev_total = total
        correct, total = update_metrics(predict_head, target_head, data["loss_mask"][:, :, None], correct, topk_acc, total)
        if accelerator.is_main_process and (idx % train_config["log_freq"] == 0) and total > prev_total and run:
            logdict = {
                "train/lr": optimizer.optimizer.param_groups[0]["lr"], 
                "train/vloss": vloss.item(),
                "train/ploss": ploss.item(), 
                "train/loss": loss.item(), 
                "train/acc": correct / total
            }
            for id, acc in enumerate(topk_acc):
                logdict[f'train/top_{id + 1}_acc'] = acc.item() / total
            run.log(logdict)
        epoch_loss += loss.item()
        num_batches += 1

    correct, total, topk_acc = gather_metrics(correct, total, topk_acc, accelerator)
    epoch_loss /= num_batches

    if accelerator.is_local_main_process and run:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        print(f'Train Accuracy: {100 * correct / total:.2f}%')
        logdict = {
            "train/epochacc": correct / total, 
            "train/epochloss": epoch_loss
        }
        for id, acc in enumerate(topk_acc):
            logdict[f'train/epochtop_{id + 1}_acc'] = acc.sum().item() / total
        run.log(logdict)


@torch.no_grad()
def validate(model, lm_head, embed_tokens, test_loader, criterion, train_config, epoch, num_epochs, save_dir, accelerator, run=None):
    model.eval()
    correct, total, epoch_loss, num_batches = 0, 0, 0, 0
    topk_acc = [0] * 3
    k_acc = [[] for _ in range(5)]

    for batch_idx, data in enumerate(tqdm(test_loader, desc="Validating")):
        if batch_idx < 10:
            acces = getkacc(model, data, lm_head, embed_tokens, max_length=5)
            for i in range(len(acces)):
                k_acc[i].append(acces[i])

        predict = model(data["hidden_states"], input_ids=data["input_ids"], embed_tokens=embed_tokens, attention_mask=data["attention_mask"])[0]
        target = data["target"]

        target_head = lm_head(target.to(lm_head.weight.dtype))
        predict_head = lm_head(predict)

        loss, ploss, vloss = calculate_loss(predict, target, predict_head, target_head, data["loss_mask"][:, :, None], criterion, train_config)

        correct, total = update_metrics(predict_head, target_head, data["loss_mask"][:, :, None], correct, topk_acc, total)
        epoch_loss += loss.item()
        num_batches += 1

    mean_acces = [torch.tensor(np.array(i).mean()).cuda() for i in k_acc]
    mean_acces = accelerator.gather_for_metrics(mean_acces)
    correct, total, topk_acc = gather_metrics(correct, total, topk_acc, accelerator)
    epoch_loss /= num_batches

    if accelerator.is_local_main_process and run:
        print(f'Test Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        print(f'Test Accuracy: {100 * correct / total:.2f}%')
        logdict = {
            "test/epochacc": correct / total, 
            "test/epochloss": epoch_loss
        }
        for id, acc in enumerate(mean_acces):
            logdict[f'test/{id}_acc'] = acc.mean().item()
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
        "num_workers": 16,
        "data_noise": True,
        "noise": "uniform",
        "mean": 0.0,
        "std": 0.2,
        "residual": "true,norm",
        # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
        "max_len": 2048,
        "grad_clip": 1.0, #0.5
        "save_freq": 5,
        "log_freq": 1#10,
    }

    # Init Accelerator
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
    testdataset = CustomDataset(testdatapath)

    train_loader = DataLoader(traindataset, batch_size=args.bs, shuffle=True,
                            collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                            pin_memory=True)
    test_loader = DataLoader(testdataset, batch_size=args.bs, shuffle=False,
                            collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)
    
    if accelerator.is_main_process:
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)


    print("Lodaing head and embed_tokens...")
    config = AutoConfig.from_pretrained(args.llm_path)
    # llm = LlamaForCausalLM.from_pretrained(
    #     args.llm_path, 
    #     torch_dtype=torch.float32,
    #     low_cpu_mem_usage=True,
    #     device_map="auto"
    # )
    llm = LlamaForCausalLM.from_pretrained(
        config=config,
        pretrained_model_name_or_path=args.llm_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    draft_config = config
    draft_config.num_hidden_layers = 1
    draft_config.use_cache = False
    draft_config._attn_implementation = "sdpa"
    
    # create new head and embed_tokens
    lm_head = torch.nn.Linear(draft_config.hidden_size, draft_config.vocab_size, bias=False)
    embed_tokens = nn.Embedding(draft_config.vocab_size, draft_config.hidden_size, draft_config.pad_token_id)
   
    # load weights
    lm_head.weight.data = llm.lm_head.weight.data
    embed_tokens.weight.data = llm.get_input_embeddings().weight.data
    
    # convert to fp32
    lm_head = lm_head.to(torch.float32)
    embed_tokens = embed_tokens.to(torch.float32)
    
    # not traininable
    for param in lm_head.parameters():
        param.requires_grad = False
    for param in embed_tokens.parameters():
        param.requires_grad = False
        
    # Before freeing up memory
    used_mem = torch.cuda.memory_allocated()
    print(f'peak mem: {used_mem / 1024 ** 3} GB')
    
    # Manually free up memory
    llm.lm_head = None
    llm.set_input_embeddings(None)
    llm.model = None
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # After freeing up memory
    used_mem = torch.cuda.memory_allocated()
    print(f'peak mem: {used_mem / 1024 ** 3} GB')
    # exit(1)


    print("Loading draft model...")
    model = DraftModel(draft_config)

    print("Setting up training...")
    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)

    # https://github.com/huggingface/diffusers/pull/6143/files
    num_update_steps_per_epoch = math.ceil(len(train_loader) / accelerator.num_processes / accelerator.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    
    num_warmup_steps = max_train_steps * args.warmup_ratio * accelerator.num_processes
    num_training_steps = max_train_steps * accelerator.num_processes
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    print("Preparing accelerator...")
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )
    
    lm_head = lm_head.to(accelerator.device)
    embed_tokens = embed_tokens.to(accelerator.device)
        
    print("Start training...")
    for epoch in range(args.epochs):
        train_one_epoch(
            model, lm_head, embed_tokens,
            train_loader, optimizer, scheduler, criterion, train_config, 
            epoch, args.epochs, accelerator, run
        )
        
        if (epoch == args.epochs-1) or (epoch % train_config["save_freq"] == 0):
            validate(
                model, lm_head, embed_tokens,
                test_loader, criterion, train_config, 
                epoch, args.epochs, args.savedir, accelerator, run
            )

    if accelerator.is_main_process:
        run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sp')
    parser.add_argument('--llm-path', '-llm', type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--datadir', type=str, default='0')
    parser.add_argument('--outdir', type=str, default='0')
    parser.add_argument('--savedir', type=str, default='0')
    parser.add_argument('--data-ratio', type=float, default=1)
    
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--warmup-ratio', type=int, default=0.05)
    parser.add_argument('--bs', type=int, default=4)
    # https://github.com/NVIDIA/NeMo/blob/876c8511e579c1c343b52bdd96ebe2296608434c/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb
    parser.add_argument('--lr', type=float, default=1e-4) #1e-2 too large, loss becomes nan when lr>3e-4
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--betas', type=float, default=(0.95, 0.5))  # from paper
    
    # logging
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    main(args)

    