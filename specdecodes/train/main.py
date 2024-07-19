import argparse
import os
import json
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from typing import Any, Dict, List
from accelerate import Accelerator
from accelerate.utils import set_seed

from ..model.configuration_eagle import EagleConfig
from ..model.ea_model import DraftModel
from ..model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from ..model.modeling_llama_kv import LlamaModel
from tqdm import tqdm
from copy import deepcopy
import wandb

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
def getkacc(model, data, head, max_length=5, dtype=torch.float32):
    hidden_states = data["hidden_states"].to(dtype)
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    target = data["target"].to(dtype)
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, sl = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    hidden_states_headout = head(hidden_states)

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
                out_hidden = model(single_hidden_states, input_ids=single_input_ids)[0]
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden.to(dtype))
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

def calculate_loss(predict, target, out_head, target_head, loss_mask, criterion, train_config):
    target_logp = nn.Softmax(dim=2)(target_head).detach()
    out_logp = nn.LogSoftmax(dim=2)(out_head)

    plogp = target_logp * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum()+1e-5)
    vloss = criterion(predict, target)
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum()+1e-5)
    loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
    return loss, ploss, vloss

@torch.no_grad()
def update_metrics(out_head, target_head, loss_mask, correct, topk_acc, total):
    _, predicted = torch.max(out_head, 2)
    _, targeted = torch.max(target_head, 2)
    ct = loss_mask.sum().item()
    cc = ((predicted == targeted) * loss_mask.squeeze()).sum().item()
    
    out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
    targeted = targeted.view(-1)[loss_mask.view(-1) == 1]

    temp_top_acc = top_accuracy(out_head, targeted, (1, 2, 3))
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


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, train_config, epoch, num_epochs, accelerator):
    model.train()
    correct, total, epoch_loss, num_batches = 0, 0, 0, 0
    topk_acc = [0] * 3

    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
        with accelerator.accumulate(model):
            optimizer.zero_grad()

            lm_head = model.module.lm_head
            predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])[0]
            target = data["target"]

            with torch.no_grad():
                target_head = lm_head(target.to(lm_head.weight.dtype))
            out_head = lm_head(predict)

            loss, ploss, vloss = calculate_loss(predict, target, out_head, target_head, data["loss_mask"][:, :, None], criterion, train_config)
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            # accelerator.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            scheduler.step()

        prev_total = total
        correct, total = update_metrics(out_head, target_head, data["loss_mask"][:, :, None], correct, topk_acc, total)
        if accelerator.is_main_process and total > prev_total:
            logdict = {
                "train/lr": optimizer.optimizer.param_groups[0]["lr"], 
                "train/vloss": vloss.item(),
                "train/ploss": ploss.item(), 
                "train/loss": loss.item(), 
                "train/acc": correct / total
            }
            for id, acc in enumerate(topk_acc):
                logdict[f'train/top_{id + 1}_acc'] = acc.item() / total
            wandb.log(logdict)
        epoch_loss += loss.item()
        num_batches += 1

    correct, total, topk_acc = gather_metrics(correct, total, topk_acc, accelerator)
    epoch_loss /= num_batches

    if accelerator.is_local_main_process:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        print(f'Train Accuracy: {100 * correct / total:.2f}%')
        logdict = {
            "train/epochacc": correct / total, 
            "train/epochloss": epoch_loss
        }
        for id, acc in enumerate(topk_acc):
            logdict[f'train/epochtop_{id + 1}_acc'] = acc.sum().item() / total
        wandb.log(logdict)


@torch.no_grad()
def validate(model, test_loader, criterion, train_config, epoch, num_epochs, save_dir, accelerator):
    model.eval()
    correct, total, epoch_loss, num_batches = 0, 0, 0, 0
    topk_acc = [0] * 3
    k_acc = [[] for _ in range(5)]

    for batch_idx, data in enumerate(tqdm(test_loader, desc="Validating")):
        if batch_idx < 10:
            acces = getkacc(model, data, model.module.lm_head, max_length=5, dtype=model.module.lm_head.weight.dtype)
            for i in range(len(acces)):
                k_acc[i].append(acces[i])

        predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])[0]
        target = data["target"]

        target_head = model.module.lm_head(target.to(model.module.lm_head.weight.dtype))
        out_head = model.module.lm_head(predict)

        loss, ploss, vloss = calculate_loss(predict, target, out_head, target_head, data["loss_mask"][:, :, None], criterion, train_config)

        correct, total = update_metrics(out_head, target_head, data["loss_mask"][:, :, None], correct, topk_acc, total)
        epoch_loss += loss.item()
        num_batches += 1

    mean_acces = [torch.tensor(np.array(i).mean()).cuda() for i in k_acc]
    mean_acces = accelerator.gather_for_metrics(mean_acces)
    correct, total, topk_acc = gather_metrics(correct, total, topk_acc, accelerator)
    epoch_loss /= num_batches

    if accelerator.is_local_main_process:
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
        wandb.log(logdict)

        # save model
        lm_head, embed_tokens = model.module.discard_head_and_embed()
        accelerator.save_model(model, f"{save_dir}/model_{epoch}")
        model.module.set_head_and_embed(lm_head, embed_tokens)


def main(args):
    set_seed(0) # fix seed

    # HUGE speedup, especially on A100 abocve
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_config = {
        "lr": args.lr,
        "bs": args.bs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "datapath": f"{args.datadir}",
        "num_epochs": 20,
        # Depending on your data and model size, the larger the model, the higher the sample efficiency. We recommend setting it between 20-40.
        "num_warmup_steps": 2000,
        "total_steps": 800000,
        "p_w": 0.1,
        "v_w": 1.0,
        "head_w": 0.1,
        "num_workers": 2,
        "embeding": True,
        "act": "No",
        "data_noise": True,
        "noise": "uniform",
        "mean": 0.0,
        "std": 0.2,
        "residual": "true,norm",
        "max_len": 2048,
        # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
        "config_path": args.configpath,
        "b1": 0.9,
        "b2": 0.95,
        "grad_clip": 0.5,
        "save_freq": 5
    }

    # Init Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=train_config["gradient_accumulation_steps"])
    
    # wandb
    if accelerator.is_main_process:
        if not args.wandb:
            os.environ['WANDB_DISABLED'] = 'true'
        wandb.init(project="eagle", config=train_config)

    # NEFTune
    if train_config["data_noise"]:
        if train_config["noise"] == "uniform":
            aug = AddUniformNoise(std=train_config["std"])
        else:
            aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
    else:
        aug = None

    # Load dataset
    datapath = list_files(train_config["datapath"])
    datapath = datapath[:int(len(datapath) * args.data_ratio)]
    print('Total data:',len(datapath))

    traindatapath = datapath[:int(len(datapath) * 0.95)]
    testdatapath = datapath[int(len(datapath) * 0.95):]

    traindataset = CustomDataset(traindatapath, transform=aug, max_len=train_config["max_len"])
    testdataset = CustomDataset(testdatapath)

    train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                            collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                            pin_memory=True)
    test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                            collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)
    
    if accelerator.is_main_process:
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)


    # Load model configs.
    config = EagleConfig.from_pretrained(train_config["config_path"])

    print("Lodaing head and embed_tokens...")
    base_model_name_or_path = config.base_model_name_or_path
    big_model = KVLlamaForCausalLM.from_pretrained(base_model_name_or_path)
    # create new head and embed_tokens
    head = torch.nn.Linear(big_model.config.hidden_size, big_model.config.vocab_size, bias=False)
    embed_tokens = nn.Embedding(big_model.config.vocab_size,big_model.config.hidden_size, big_model.config.pad_token_id)
    # not traininable
    for param in head.parameters():
        param.requires_grad = False
    for param in embed_tokens.parameters():
        param.requires_grad = False
    # delete big model
    del big_model

    print("Loading draft model...")
    draft_config = deepcopy(config)
    draft_config.num_hidden_layers = 1
    tiny_model = LlamaModel(draft_config)
    model = DraftModel(draft_config)
    model.model = tiny_model
    model.lm_head = head
    model.embed_tokens = embed_tokens

    print("Loaded.")


    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

    num_epochs = train_config["num_epochs"]
    num_warmup_steps = train_config["num_warmup_steps"]
    total_steps = train_config["total_steps"]

    # get_linear_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )
        
    print("Start training...")
    for epoch in range(num_epochs):
        train_one_epoch(
            model, train_loader, 
            optimizer, scheduler, criterion, train_config, 
            epoch, num_epochs, accelerator
        )
        
        if (epoch == num_epochs-1) or (epoch % train_config["save_freq"] == 0):
            validate(
                model, test_loader, 
                criterion, train_config, 
                epoch, num_epochs, args.savedir, accelerator
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sp')
    parser.add_argument('--configpath', type=str, default="config.json")
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--datadir', type=str, default='0')
    parser.add_argument('--outdir', type=str, default='0')
    parser.add_argument('--savedir', type=str, default='0')
    parser.add_argument('--data-ratio', type=float, default=1)
    # set to true without having to pass a true argument
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    main(args)

    