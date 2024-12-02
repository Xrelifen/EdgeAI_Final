import os
import logging
import argparse
from copy import deepcopy

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, AutoConfig, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed, release_memory, tqdm
from accelerate.logging import get_logger
import wandb

from .liger_mokeypatch import apply_liger_kernel_to_llama
from ..models import SSM_Eagle, LLM_First_Layers, LLM_Last_Layers, SSM_Custom
from .train_utils import (
    list_files,
    CustomDataset,
    DataCollatorWithPadding,
    AddUniformNoise,
    AddGaussianNoise,
    update_metrics,
    calc_sl1_loss,
    calc_ce_loss,
    wsd_schedule,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

def calculate_loss(loss_mask, s_hidden_states, t_hidden_states, s_logits, t_logits, train_config):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    s_logits = s_logits.float()
    t_logits = t_logits.float()
    
    # Calculate losses
    # vloss = calc_sl1_loss(loss_mask, s_hidden_states, t_hidden_states)
    # ploss = calc_ce_loss(loss_mask, s_logits, t_logits)
    # loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
    # return loss, vloss, ploss
    
    ploss = calc_ce_loss(loss_mask, s_logits, t_logits)
    return ploss, ploss, ploss

def train_one_epoch(
    model, llm_first, llm_last, train_loader, optimizer, scheduler,
    train_config, epoch, num_epochs, accelerator, run=None
):
    model.train()
    total_correct = 0
    total_samples = 0
    total_expect = 0
    total_loss = 0
    total_topk_metrics = {f"prob@{k}": 0.0 for k in [1, 3, 5, 9]}
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=not accelerator.is_local_main_process)
    for step, data in enumerate(progress_bar):
        with accelerator.accumulate(model):
            # Teacher outputs
            with torch.no_grad():
                t_hidden_states = data["target"]
                t_logits = llm_last(t_hidden_states)
                
            # Student outputs
            s_logits, s_hidden_states = model(
                input_ids=data["input_ids"],
                hidden_states=data["hidden_states"],
                attention_mask=data["attention_mask"],
                return_logits=True,
            )

            # Calculate loss
            loss, vloss, ploss = calculate_loss(
                data["loss_mask"], s_hidden_states, t_hidden_states,
                s_logits, t_logits, train_config
            )

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update metrics
        correct, total, expect, topk_metrics = update_metrics(
            loss_mask=data["loss_mask"],
            s_logits=s_logits,
            t_logits=t_logits,
        )

        # Logging
        if accelerator.is_main_process and step % train_config["log_freq"] == 0 and run:
            current_lr = scheduler.get_last_lr()[0]
            log_dict = {
                "train/lr": current_lr,
                "train/loss": loss.item(),
                "train/v_loss": vloss.item(),
                "train/p_loss": ploss.item(),
                "train/accuracy": correct / total if total > 0 else 0,
                "train/expect_value": expect / total if total > 0 else 0,
            }
            for k, v in topk_metrics.items():
                log_dict[f"train/{k}"] = v / total if total > 0 else 0
            run.log(log_dict)
    
        # Accumulate metrics for epoch
        total_correct += correct
        total_samples += total
        total_expect += expect
        total_loss += loss.item()
        for k, v in topk_metrics.items():
            total_topk_metrics[k] += v

    # Epoch logging
    if accelerator.is_main_process and run:
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_expect = total_expect / total_samples if total_samples > 0 else 0
        
        logger.info(
            f'Epoch [{epoch + 1}/{num_epochs}] ' 
            f'Loss: {avg_loss:.4f}, '
            f'Accuracy: {100 * correct / total:.2f}%'
        )
        log_dict = {
            "train/epoch_loss": avg_loss,
            "train/epoch_accuracy": avg_accuracy,
            "train/epoch_expect_value": avg_expect,
        }
        for k, v in total_topk_metrics.items():
            log_dict[f"train/epoch_{k}"] = v / total_samples if total_samples > 0 else 0
        run.log(log_dict)
        
    accelerator.wait_for_everyone()

@torch.no_grad()
def validate(
    model, llm_first, llm_last, test_loader, train_config,
    epoch, num_epochs, save_dir, accelerator, run=None
):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_expect = 0
    total_loss = 0
    total_ploss = 0
    total_vloss = 0
    total_topk_metrics = {f"prob@{k}": 0.0 for k in [1, 3, 5, 9]}

    progress_bar = tqdm(test_loader, desc="Validating", disable=not accelerator.is_local_main_process)
    for data in progress_bar:
        # Teacher outputs
        t_hidden_states = data["target"]
        t_logits = llm_last(t_hidden_states)

        # Student outputs
        s_logits, s_hidden_states = model(
            input_ids=data["input_ids"],
            hidden_states=data["hidden_states"],
            attention_mask=data["attention_mask"],
            return_logits=True,
        )

        # Calculate loss
        loss, vloss, ploss = calculate_loss(
            data["loss_mask"], s_hidden_states, t_hidden_states,
            s_logits, t_logits, train_config
        )

        # Update metrics
        correct, total, expect, topk_metrics = update_metrics(
            loss_mask=data["loss_mask"],
            s_logits=s_logits,
            t_logits=t_logits,
        )
        total_correct += correct
        total_samples += total
        total_expect += expect
        for k, v in topk_metrics.items():
            total_topk_metrics[k] += v
        total_loss += loss.item()
        total_ploss += ploss.item()
        total_vloss += vloss.item()
        
    # gather metrics
    device = accelerator.device
    total_correct = accelerator.reduce(torch.tensor(total_correct, device=device)).sum().item()
    total_samples = accelerator.gather_for_metrics(torch.tensor(total_samples, device=device)).sum().item()
    total_expect = accelerator.gather_for_metrics(torch.tensor(total_expect, device=device)).sum().item()
    avg_loss = accelerator.gather_for_metrics(torch.tensor(total_loss/len(test_loader), device=device)).mean().item()
    avg_ploss = accelerator.gather_for_metrics(torch.tensor(total_ploss/len(test_loader), device=device)).mean().item()
    avg_vloss = accelerator.gather_for_metrics(torch.tensor(total_vloss/len(test_loader), device=device)).mean().item()
    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_expect = total_expect / total_samples if total_samples > 0 else 0
    reduce_topk_metrics = {}
    for k, v in total_topk_metrics.items():
        v = torch.tensor(v, device=device)
        v = accelerator.gather_for_metrics(v)
        reduce_topk_metrics[k] = v.sum().item() / total_samples if total_samples > 0 else 0 

    # Validation logging
    if accelerator.is_main_process and run:
        logger.info(
            f'Validation Epoch [{epoch + 1}/{num_epochs}] '
            f'Loss: {avg_loss:.4f}, '
            f'Accuracy: {100 * total_correct / total_samples:.2f}%'
        )
        log_dict = {
            "val/accuracy":avg_accuracy,
            "val/expect_value": avg_expect,
            "val/loss": avg_loss,
            "val/v_loss": avg_vloss,
            "val/p_loss": avg_ploss,
        }
        for k, v in reduce_topk_metrics.items():
            log_dict[f"val/{k}"] = v
        run.log(log_dict)

        # Save model checkpoint
        if (epoch + 1) % train_config["save_freq"] == 0 or (epoch + 1) == num_epochs:
            save_path = os.path.join(save_dir, f"model_{epoch + 1}")
            accelerator.save_model(model, save_path)
            logger.info(f"Model saved at: {save_path}")
            
    accelerator.wait_for_everyone()
            
def main(args):
    # Enable TF32 for speedup on compatible GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(0)  # Fix seed for reproducibility

    train_config = {
        "p_w": 0.1,
        "v_w": 1.0,
        "num_workers": 4,
        "data_noise": True,
        "noise": "uniform",
        "mean": 0.0,
        "std": 0.2,
        "max_len": 2048,
        "grad_clip": 0.5,
        "save_freq": 5,
        "log_freq": 10,
    }
    train_config.update(vars(args))

    # Initialize Accelerator
    accelerator = Accelerator()
    logger.info(f"Device: {accelerator.device}")

    # Initialize wandb
    if accelerator.is_main_process:
        if not args.wandb:
            os.environ['WANDB_DISABLED'] = 'true'
        wandb.require("core")
        run = wandb.init(project=args.wandb_project, config=train_config)
    else:
        run = None

    # Data augmentation
    aug = None
    if train_config["data_noise"]:
        if train_config["noise"] == "uniform":
            aug = AddUniformNoise(std=train_config["std"])
        else:
            aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])

    # Load dataset
    datapath = list_files(args.datadir)
    datapath = datapath[:int(len(datapath) * args.data_ratio)]
    logger.info(f'Total data files: {len(datapath)}')

    split_idx = int(len(datapath) * 0.95)
    traindatapath = datapath[:split_idx]
    testdatapath = datapath[split_idx:]

    traindataset = CustomDataset(traindatapath, transform=aug, max_len=train_config["max_len"])
    testdataset = CustomDataset(testdatapath, max_len=train_config["max_len"])

    data_collator = DataCollatorWithPadding()
    train_loader = DataLoader(
        traindataset,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=train_config["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        testdataset,
        batch_size=args.bs,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=train_config["num_workers"],
        pin_memory=True
    )

    # Ensure save directory exists
    if accelerator.is_main_process:
        os.makedirs(args.savedir, exist_ok=True)

    # Load LLM model and configurations
    logger.info(f"Loading LLM partial layers...")
    config = AutoConfig.from_pretrained(args.llm_path)
    llm = LlamaForCausalLM.from_pretrained(
        args.llm_path,
        config=config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    # Extract first and last layers
    llm_first = LLM_First_Layers(llm, keep_layers_num=args.llm_first_layers)
    llm_last = LLM_Last_Layers(llm, keep_layers_num=args.llm_last_layers)
    llm_first.eval().to(accelerator.device)
    llm_last.eval().to(accelerator.device)

    # Set up the student model
    logger.info("Setting up model...")
    draft_config = deepcopy(config)
    draft_config.num_hidden_layers = 1
    # draft_config.head_dim = 64
    # draft_config.hidden_size = 2048
    # draft_config.intermediate_size = 8192
    # draft_config.num_attention_heads = 32
    # draft_config.num_key_value_heads = 8
    
    # draft_config.hidden_size = 576
    # draft_config.intermediate_size = 1536
    # draft_config.hidden_size = 4096 // 4
    # draft_config.intermediate_size = 11008 // 4
    
    
    draft_config.use_cache = False
    draft_config._attn_implementation = "sdpa"
    if args.neftune:
        draft_config.neftune_noise_alpha = args.neftune_noise_alpha

    if args.pretrained:
        logger.info("Loading pretrained model...")
        model = SSM_Custom.from_pretrained(args.pretrained, config=draft_config, keep_embeddings=args.keep_embeddings)
    else:
        logger.info("Loading draft model...")
        model = SSM_Custom(config=draft_config, keep_embeddings=args.keep_embeddings)

    # apply liger kernel to draft model
    apply_liger_kernel_to_llama(model=model.model, rms_norm=False)

    # Transfer embeddings if available
    # if hasattr(model.model, "embed_tokens"):
    #     model.model.embed_tokens.weight.data = llm.get_input_embeddings().weight.data.clone()
    #     model.model.embed_tokens.requires_grad_(False)
        
    # if hasattr(model, "lm_head"):
    #     logger.info("Transferring lm_head...")
    #     model.lm_head.weight.data = llm.lm_head.weight.data.clone()
    #     model.lm_head.weight.requires_grad_(False)

    # # load llm's norm layer to draft model
    # model.model.norm.weight = llm.model.norm.weight.clone().detach()
    # model.model.norm.requires_grad_(False)
    
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


    # Clean up to free memory
    del llm
    release_memory()

    # Calculate training steps
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    logger.info(f'Warmup steps: {num_warmup_steps}, Total training steps: {max_train_steps}')

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=max_train_steps
    # )
    wsd_fract_decay = 0.2
    lambda_schedule = wsd_schedule(
        n_iterations=max_train_steps,
        n_warmup=num_warmup_steps,
        fract_decay=wsd_fract_decay,
        init_div_factor=1e2,
        final_lr_factor=0,  # should be 0 here
        decay_type='linear',
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_schedule)

    # Prepare everything with accelerator
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )

    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        # Train
        train_one_epoch(
            model, llm_first, llm_last, train_loader, optimizer, scheduler,
            train_config, epoch, args.epochs, accelerator, run
        )

        # Validate and save model
        if not args.no_validate:
            validate(
                model, llm_first, llm_last, test_loader, train_config,
                epoch, args.epochs, args.savedir, accelerator, run
            )

    # Finish wandb run
    if accelerator.is_main_process and run:
        run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--llm-path', type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--data-ratio', type=float, default=1.0)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--no-validate', action='store_true', help='Skip validation')

    # Model parameters
    parser.add_argument('--keep-embeddings', action='store_true', help='Keep embeddings saved in the model')
    parser.add_argument('--llm-first-layers', type=int, default=0)
    parser.add_argument('--llm-last-layers', type=int, default=0)

    # Training parameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--betas', nargs=2, type=float, default=(0.9, 0.95))
    parser.add_argument('--neftune', action='store_true')
    parser.add_argument('--neftune-noise-alpha', type=float, default=5.0)

    # Logging
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='specdecodes')
    args = parser.parse_args()
    main(args)
