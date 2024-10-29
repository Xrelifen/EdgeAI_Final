import argparse
from copy import deepcopy
import time
import torch
import random
import os
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer#, AutoModelForCausalLM
from fastchat.utils import str_to_torch_dtype
from ..models import HuggingFaceWrapper, ProfileNaiveWrapper, NaiveWrapper, SDWrapper, ProfileSDWrapper, SSM_Classic, SSM_Eagle, ProfileOffloadSDWrapper
from ..models import modeling_llama

# Set random seed for reproducibility
torch.manual_seed(0)
random.seed(0)

def load_model(llm_path, ssm_path, mode, sd_method, layers, llm_offload=False, out_dir=None, dtype=torch.float16, device="auto"):
    # Load tokenizer and LLM
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
    need_offload = "offload" in mode

    assert need_offload == llm_offload, "Offload Mode needs to be set when offload is included in sd-method" 

    llm = None
    
    if not llm_offload:
        llm = modeling_llama.LlamaForCausalLM.from_pretrained(llm_path, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=device)

        # Prepare SSM configuration
        draft_config = deepcopy(llm.config) if os.path.exists(ssm_path) else None
        if draft_config:
            draft_config.num_hidden_layers = layers
    else:
        draft_config = None

    # Select model wrapper
    wrappers = {
        "naive": ProfileNaiveWrapper, # NaiveWrapper,
        "hf": HuggingFaceWrapper,
        "sd-classic": lambda: ProfileSDWrapper(out_dir=out_dir),
        "sd-eagle": lambda: ProfileSDWrapper(out_dir=out_dir),
        "sd-classic-offload": lambda: ProfileOffloadSDWrapper(out_dir=out_dir),
    }
    model = wrappers.get(mode, lambda: ValueError("Invalid mode"))()

    # Load SSM if required
    if mode.startswith("sd"):
        ssm_cls = SSM_Classic if "sd-classic" in mode else SSM_Eagle
        ssm = ssm_cls.from_pretrained(
            ssm_path, 
            config=draft_config, 
            sampling_method=sd_method,
            eos_token_id=tokenizer.eos_token_id,
            tree_depth=12,
            topk_len=16,
            min_sample_prob=1e-2,
            min_accept_prob=1e-2,
            torch_dtype=dtype
        )

        if llm is not None:
            ssm = ssm.to(llm.model.layers[-1].self_attn.q_proj.weight.device)
        else:
            ssm = ssm.to(device)

        ssm = torch.compile(ssm, mode="reduce-overhead")
        model.set_ssm(ssm)

    if llm_offload: 
        model.set_offload_llm(llm_path, device=device)
    else:
        model.set_llm(llm)

    model.set_tokenizer(tokenizer)
    model.eval()

    return model, tokenizer

def run_eval(llm_path, ssm_path, mode, sd_method, layers, out_dir, dataset, log_file,
             max_new_tokens, llm_offload=False, temp=0.6, dtype=torch.float16, do_sample=False):
    model, tokenizer = load_model(llm_path, ssm_path, mode, sd_method, layers, llm_offload, out_dir, dtype=dtype, device="cuda")

    # Warm up the model
    warmup_input = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello."}]
    input_ids = tokenizer.apply_chat_template(warmup_input, tokenize=True, add_generation_prompt=True,
                                              return_tensors="pt").cuda()
    model.generate(input_ids, temperature=temp, max_new_tokens=max_new_tokens, do_sample=do_sample)

    tput_list, accept_rate_list = [], []

    # Evaluate dataset
    for idx, query in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating"):
        input_message = query.replace("[INST]", "").replace("[/INST]\n\nASSISTANT:", "")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_message}
        ]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  return_tensors="pt").cuda()
        output_ids = model.generate(input_ids, temperature=temp, max_new_tokens=max_new_tokens, do_sample=do_sample)
        output = tokenizer.decode(output_ids[0][input_ids.shape[1]:])

        exp_log = {**model.exp_log, "query": input_message, "response": output}
        with open(log_file, 'a+') as f:
            json.dump(exp_log, f, indent=4)
            f.write("\n")

        tput_list.append(exp_log.get("tput", 0))
        if "sd" in mode:
            accept_rate_list.append(exp_log.get("avg_sampled", 0))

    avg_tput = np.mean(tput_list)
    avg_accept_rate = np.mean(accept_rate_list) if accept_rate_list else 0
    
    return avg_tput, avg_accept_rate    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-path", "-llm", required=True, help="Path to LLM weights.")
    parser.add_argument("--ssm-path", "-ssm", default="", help="Path to SSM weights.")
    parser.add_argument("--mode", default="naive", help="Model mode.")
    parser.add_argument("--sd-method", default="greedy", help="Sampling method for SD.")
    parser.add_argument("--layers", type=int, default=1, help="Number of SSM layers.")
    parser.add_argument("--out-dir", default="specdecodes/experiments/mt_bench/", help="Output directory.")
    parser.add_argument("--log-dir", default="specdecodes/experiments/result/", help="Experiment log directory.")
    parser.add_argument("--bench-name", default="mt_bench", help="Benchmark name.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max new tokens.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--repeat", type=int, default=3, help="Repeat evaluation.")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for sampling.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--llm_offload", action="store_true", help="Offload LLM")

    args = parser.parse_args()

    if args.bench_name != "mt_bench":
        raise NotImplementedError("Only 'mt_bench' dataset is supported.")

    with open("specdecodes/benchmark/data/mt_bench.json") as f:
        dataset = [x[1] for x in json.load(f)]
        random.shuffle(dataset)

    os.makedirs("specdecodes/experiments/result", exist_ok=True)
    llm_name, ssm_name = os.path.basename(args.llm_path), os.path.basename(args.ssm_path)

    # Handle output directories
    if args.out_dir:
        os.system(f"rm -rf {args.out_dir}")
        print(f"Deleted old {args.out_dir}")
    log_dir = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}")
    print(f"Log directory: {log_dir}")
    
    tput_list, accept_rate_list = [], []
    for i in range(args.repeat):
        log_file = os.path.join(log_dir, f"{i}.jsonl")
        avg_tput, avg_accept_rate = run_eval(
            args.llm_path, args.ssm_path, args.mode, args.sd_method, 
            args.layers, args.out_dir, dataset, log_file, args.max_new_tokens, 
            args.llm_offload, args.temp, str_to_torch_dtype(args.dtype), args.do_sample
        )
        tput_list.append(avg_tput)
        accept_rate_list.append(avg_accept_rate)
    
    tput_mean, tput_std = np.mean(tput_list), np.std(tput_list)
    accept_rate_mean, accept_rate_std = np.mean(accept_rate_list), np.std(accept_rate_list)
    print(f"Throughput: {tput_mean:.2f} ± {tput_std:.2f} tokens/sec")
    if accept_rate_mean > 0:
        print(f"Acceptance rate: {accept_rate_mean:.2f} ± {accept_rate_std:.2f}")
        
