import math
import os
from copy import deepcopy

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed, release_memory, tqdm
from accelerate.logging import get_logger
import logging
import argparse
import wandb

from liger_kernel.transformers import apply_liger_kernel_to_llama
from ..models import SSM_Classic, LLM_First_Layers, LLM_Last_Layers
from .train_utils import *

# logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def calculate_loss(loss_mask, s_hidden_states, t_hidden_states, s_logits, t_logits, train_config):
    vloss = calc_smooth_l1_loss(loss_mask, s_hidden_states, t_hidden_states)
    ploss = calc_ce_loss(loss_mask, s_logits, t_logits)
    loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
    return loss, vloss, ploss
    # return ploss, ploss, ploss

def train_one_epoch(model, llm_first, llm_last, train_loader, optimizer, scheduler, train_config, epoch, num_epochs, accelerator, run=None):
    model.train()
    device = accelerator.device
    correct, total, epoch_loss, num_batches = 0, 0, 0, 0
    topk_prob_k = [1, 3, 5, 9]
    topk_prob = [0] * len(topk_prob_k)

    for idx, data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
        with accelerator.accumulate(model):
            # teacher
            with torch.no_grad():
                t_hidden_states = data["target"] # data["target"] equals data["hidden_states"][:, 1:]
                t_logits = llm_last(t_hidden_states)
            
            # student
            s_hidden_states = model(input_ids=data["input_ids"], embed_tokens=llm_first, attention_mask=data["attention_mask"])[0]
            s_logits = llm_last(s_hidden_states, head_only=True)
            
            # Calculate loss
            loss, vloss, ploss = calculate_loss(data["loss_mask"], s_hidden_states, t_hidden_states, s_logits, t_logits, train_config)
            
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        prev_total = total
        expect, correct, total = update_metrics(data["loss_mask"], s_logits, t_logits, correct, total, topk_prob)
        if accelerator.is_main_process and (idx % train_config["log_freq"] == 0) and total > prev_total and run:
            logdict = {
                "train/lr": optimizer.optimizer.param_groups[0]["lr"], 
                "train/vloss": vloss.item(),
                "train/ploss": ploss.item(), 
                "train/loss": loss.item(), 
                "train/acc": correct / total,
                "train/expect": expect.item()
            }
            for id, prob in enumerate(topk_prob):
                logdict[f'train/prob@{topk_prob_k[id]}'] = prob.item() / total
            run.log(logdict)
        epoch_loss += loss.item()
        num_batches += 1

    # gather metrics
    correct, total = torch.tensor(correct, device=device), torch.tensor(total, device=device)
    correct, total = accelerator.gather_for_metrics((correct, total))
    expect = accelerator.gather_for_metrics(expect)
    topk_prob = accelerator.gather_for_metrics(topk_prob)
    
    expect = expect.mean().item()
    correct = correct.sum().item()
    total = total.sum().item()
    epoch_loss /= num_batches

    if accelerator.is_local_main_process and run:
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}]\nLoss: {epoch_loss:.4f}, Train Accuracy: {100 * correct / total:.2f}%')
        logdict = {
            "train/epochacc": correct / total, 
            "train/epochloss": epoch_loss,
            "train/epochexpect": expect,
        }
        for id, prob in enumerate(topk_prob):
            logdict[f'train/epochprob@{topk_prob_k[id]}'] = prob.sum().item() / total
        run.log(logdict)


@torch.no_grad()
def validate(model, llm_first, llm_last, test_loader, train_config, epoch, num_epochs, save_dir, accelerator, run=None):
    device = accelerator.device
    model.eval()
    correct, total, epoch_loss, num_batches = 0, 0, 0, 0
    topk_prob_k = [1, 3, 5, 9]
    topk_prob = [0] * len(topk_prob_k)

    for batch_idx, data in enumerate(tqdm(test_loader, desc="Validating")):
        # teacher
        t_hidden_states = data["target"] # data["target"] equals data["hidden_states"][:, 1:]
        t_logits = llm_last(t_hidden_states)
        
        # student
        s_hidden_states = model(input_ids=data["input_ids"], embed_tokens=llm_first, attention_mask=data["attention_mask"])[0]
        s_logits = llm_last(s_hidden_states, head_only=True)

        # Calculate loss
        loss, vloss, ploss = calculate_loss(data["loss_mask"], s_hidden_states, t_hidden_states, s_logits, t_logits, train_config)

        # Update metrics
        expect, correct, total = update_metrics(data["loss_mask"], s_logits, t_logits, correct, total, topk_prob)
        epoch_loss += loss.item()
        num_batches += 1

    # gather metrics
    correct, total = torch.tensor(correct, device=device), torch.tensor(total, device=device)
    correct, total = accelerator.gather_for_metrics((correct, total))
    expect = accelerator.gather_for_metrics(expect)
    topk_prob = accelerator.gather_for_metrics(topk_prob)
    
    expect = expect.mean().item()
    correct = correct.sum().item()
    total = total.sum().item()
    epoch_loss /= num_batches

    # Log and save model
    if accelerator.is_local_main_process and run:
        logger.info(f'Test Epoch [{epoch + 1}/{num_epochs}]\n Loss: {epoch_loss:.4f}, Test Accuracy: {100 * correct / total:.2f}%')
        logdict = {
            "test/epochacc": correct / total, 
            "test/epochloss": epoch_loss,
            "test/ploss": ploss.item(),
            "test/vloss": vloss.item(),
            "test/expect": expect,
        }

        for id, prob in enumerate(topk_prob):
            logdict[f'test/prob@{topk_prob_k[id]}'] = prob.sum().item() / total
        run.log(logdict)

        # save model
        # use path join to avoid os error
        save_location = os.path.join(save_dir, f"model_{epoch + 1}")
        accelerator.save_model(model, save_location)
        

def main(args):
    # HUGE speedup, especially on A100 or above
    torch.backends.cuda.matmul.allow_tf32 = True
    # fix seed
    set_seed(0)

    train_config = {
        "p_w": 0.1,
        "v_w": 1.0,
        "num_workers": 4,
        # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
        "max_len": 2048,
        "grad_clip": 0.5,
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

    # Load dataset
    datapath = list_files(args.datadir)
    datapath = datapath[:int(len(datapath) * args.data_ratio)]
    logger.info(f'Total data: {len(datapath)}')

    traindatapath = datapath[:int(len(datapath) * 0.95)]
    testdatapath = datapath[int(len(datapath) * 0.95):]

    traindataset = CustomDataset(traindatapath, max_len=train_config["max_len"])
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
    logger.info("Loading head...")
    config = AutoConfig.from_pretrained(args.llm_path)
    llm = LlamaForCausalLM.from_pretrained(
        config=config,
        pretrained_model_name_or_path=args.llm_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        # device_map="auto"
    )
    
    # Load llm's first and last layers
    llm_first = LLM_First_Layers(llm, keep_layers_num=args.llm_first_layers) # embed_tokens
    llm_last = LLM_Last_Layers(llm, keep_layers_num=args.llm_last_layers) # lm_head
    llm_first.eval()
    llm_last.eval()
    
    # Set draft model config
    draft_config = deepcopy(llm.config)
    draft_config.num_hidden_layers = args.layers
    draft_config.use_cache = False
    draft_config._attn_implementation = "sdpa"
    if args.neftune:
        draft_config.neftune_noise_alpha = args.neftune_noise_alpha
    
    # load weights from pretrained model if specified
    if args.pretrained is not None:
        logger.info("Loading pretrained model...")
        model = SSM_Classic.from_pretrained(args.pretrained, config=draft_config, keep_embeddings=args.keep_embeddings)
    else:
        logger.info("Loading draft model...")
        model = SSM_Classic(config=draft_config, keep_embeddings=args.keep_embeddings)
        
    # ----------------- Load llm's weights to draft model -----------------
    
    # load llm's embeddings to draft model if exists
    if getattr(model.model, "embed_tokens", None) is not None:
        for param, llm_param in zip(model.model.embed_tokens.parameters(), llm.get_input_embeddings().parameters()):
            param.data = llm_param.data
            param.requires_grad = False

    # # load llm's norm layer to draft model
    # model.model.norm.weight.data = llm.model.norm.weight.data
    # model.model.norm.weight.requires_grad = False
    
    # # load llm's last attention layer's data to draft model (not trainable)
    # load_index = -1
    # for (draft_param, llm_param) in zip(model.model.layers[load_index].parameters(), llm.model.layers[load_index].parameters()):
    #     draft_param.data = llm_param.data
    #     draft_param.requires_grad = False
    
    # # load llm's first layer's data to draft model
    # load_index = -2
    # for (draft_param, llm_param) in zip(model.model.layers[load_index].parameters(), llm.model.layers[load_index].parameters()):
    #     draft_param.data = llm_param.data
    #     draft_param.requires_grad = False
    
    # ----------------- Load llm's weights to draft model -----------------
        
    # apply liger kernel to ssm model
    # apply_liger_kernel_to_llama(model=llm_last)
    apply_liger_kernel_to_llama(model=model.model, rms_norm=False) # eagle removes some norm layers, so rms_norm=False
    
    # Manually free up memory by deleting llm
    logger.info(f'Current used GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3} GB')
    del llm
    release_memory()
    logger.info(f'GPU memory after freeing llm: {torch.cuda.memory_allocated() / 1024 ** 3} GB')

    # Calculate the number of update steps per epoch  https://github.com/huggingface/diffusers/pull/6143/files
    logger.info("Setting up training...")
    num_update_steps_per_epoch = math.ceil(len(train_loader) / accelerator.num_processes / accelerator.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    num_warmup_steps = max_train_steps * args.warmup_ratio * accelerator.num_processes
    num_training_steps = max_train_steps * accelerator.num_processes
    logger.info(f'warmup steps: {num_warmup_steps}, training steps: {num_training_steps}')
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    # Prepare accelerator
    logger.info("Preparing accelerator...")
    train_loader, test_loader, accelerate_model, llm_first, llm_last, optimizer, scheduler = accelerator.prepare(
        train_loader, test_loader, model, llm_first, llm_last, optimizer, scheduler
    )  
 
    # Training loop
    logger.info("Start training...")
    for epoch in range(args.epochs):
        # Train
        model.activate_forward_hooks()
        train_one_epoch(
            accelerate_model, llm_first, llm_last,
            train_loader, optimizer, scheduler, train_config, 
            epoch, args.epochs, accelerator, run
        )
        model.deactivate_forward_hooks()
        
        # Validate
        if not args.no_validate:
            if (epoch == args.epochs-1) or (epoch % train_config["save_freq"] == 0):
                validate(
                    accelerate_model, llm_first, llm_last,
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
    parser.add_argument('--no-validate', '-nv', action='store_true', help='Skip validation')
    
    # model parameters
    parser.add_argument('--keep-embeddings', '-le', action='store_true', help='Keep embeddings saved in the model')
    parser.add_argument('--llm-first-layers', '-fl', type=int, default=0)
    parser.add_argument('--llm-last-layers', '-ll', type=int, default=0)
    
    # training parameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--warmup-ratio', type=int, default=0.05)
    parser.add_argument('--bs', type=int, default=4)
    # https://github.com/NVIDIA/NeMo/blob/876c8511e579c1c343b52bdd96ebe2296608434c/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb
    parser.add_argument('--lr', type=float, default=1e-4) #1e-2 too large, loss becomes nan when lr>3e-4
    parser.add_argument('--weight-decay', type=float, default=1e-2) # from paper: 1e-3
    parser.add_argument('--betas', type=float, default=(0.9, 0.95))  # from paper: (0.95, 0.5)
    # neftune (currently seems to decrease performance on eagle)
    parser.add_argument('--neftune', action='store_true')
    parser.add_argument('--neftune-noise-alpha', '-nefta', type=float, default=5)
    
    # logging
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    main(args)

    