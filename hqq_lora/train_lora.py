import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.core.quantize import *
from hqq.core.peft import PeftUtils
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from utils import get_quantized_model
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def train_lora(model, tokenizer, device='cuda'):
    # train_dtype       = torch.float32
    # atten_lora_params = {'lora_type':'default', 'r':32, 'lora_alpha':32, 'dropout':0.05, 'train_dtype':train_dtype, 'train_bias':True}
    # mlp_lora_params   = {'lora_type':'default', 'r':8,  'lora_alpha':8,  'dropout':0.05, 'train_dtype':train_dtype, 'train_bias':True}

    # lora_params       = {'self_attn.q_proj': atten_lora_params,
    #                     'self_attn.k_proj': atten_lora_params,
    #                     'self_attn.v_proj': atten_lora_params,
    #                     'self_attn.o_proj': atten_lora_params,
    #                     'mlp.gate_proj'   : mlp_lora_params,
    #                     'mlp.up_proj'     : mlp_lora_params,
    #                     'mlp.down_proj'   : mlp_lora_params}

    # PeftUtils.add_lora(model, lora_params)
    
    # HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
    
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True,
    )
    model = get_peft_model(model, config)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side  = "right" 
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    
    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            max_seq_length=256,
            packing=True,
            output_dir=f"./results_lora_1B",
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            # optim="paged_adamw_32bit",
            # logging_dir="./logs",
            save_steps=5000,
            logging_steps=1,
            learning_rate=1e-5,
            remove_unused_columns=False,
            # weight_decay=0,
            # max_grad_norm=0.3,
            max_steps=-1,
            # warmup_ratio=0.03,
            fp16=True,
            # group_by_length=True,
            lr_scheduler_type="cosine",
        ),
    )
    model.config.use_cache = False
    model.is_parallelizable       = False
    trainer.is_model_parallel     = False
    trainer.place_model_on_device = False
    model.train()
    
    trainer.train()

if __name__ == "__main__":
    device = 'cuda'

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    # get_quantized_model(model=model, device=device)
    
    # model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
    
    train_lora(model, tokenizer, device=device)