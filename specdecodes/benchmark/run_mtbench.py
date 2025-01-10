import argparse
from copy import deepcopy
import time
import torch
import random
import os
import json
import numpy as np
from tqdm import tqdm, trange
from transformers import AutoTokenizer#, AutoModelForCausalLM
from fastchat.utils import str_to_torch_dtype

from torch.nn.attention import SDPBackend, sdpa_kernel
from ..models import HuggingFaceWrapper, ProfileNaiveWrapper, NaiveWrapper, SDWrapper, ProfileSDWrapper, SSM_Classic, SSM_Eagle
from ..models import DraftParams, modeling_llama

# Set random seed for reproducibility
torch.manual_seed(0)
random.seed(0)

def load_model(
    llm_path: str,
    ssm_path: str,
    out_dir=None,
    dtype: torch.dtype = torch.float16,
    device: str = "auto",
    args: dict = {},
    ):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
    
    # load LLM
    # llm = AutoModelForCausalLM.from_pretrained(
    llm = modeling_llama.LlamaForCausalLM.from_pretrained(
        llm_path, 
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device,
        _attn_implementation="sdpa",
    )
    ssm = None
    
    # check if ssm_path directory exists
    if os.path.exists(ssm_path):
        draft_config = deepcopy(llm.config)
        draft_config.num_hidden_layers = 1
        
    else:
        draft_config = None

    if args.mode == "naive":
        model = ProfileNaiveWrapper()
        
    elif args.mode == "hf":
        model = HuggingFaceWrapper() # should not work
        
    elif args.mode.split("-")[0] == "sd":
        if args.mode == "sd-classic":
            # load SSM
            ssm = SSM_Classic.from_pretrained(
                ssm_path,
                config=draft_config,
                eos_token_id=tokenizer.eos_token_id,
                torch_dtype=dtype,
            ).to(llm.model.layers[-1].self_attn.q_proj.weight.device)
        elif args.mode == "sd-eagle":
            # load SSM
            ssm = SSM_Eagle.from_pretrained(
                ssm_path,
                config=draft_config,
                eos_token_id=tokenizer.eos_token_id,
                torch_dtype=dtype,
                keep_embeddings=False,
            ).to(llm.model.layers[-1].self_attn.q_proj.weight.device)
            ssm.set_modules(embed_tokens=llm.get_input_embeddings(), lm_head=llm.lm_head)
        else:
            raise ValueError("Invalid sd mode.")

        draft_params = DraftParams(
            max_depth=args.max_depth,
            topk_len=10,
            min_accept_prob=0.01
        )
        print("Draft params:", draft_params)
        
        model = ProfileSDWrapper(draft_params=draft_params, out_dir=out_dir)
        model.set_ssm(ssm)
        
    else:
        raise ValueError("Invalid mode.")

    # set model
    model.cache_implementation = args.cache_impl
    model.set_tokenizer(tokenizer)
    model.set_llm(llm)
    model.eval()
    llm.eval()
    ssm.eval()
    
    if args.compile_mode != 'eager':
        print("Running with Torch Inductor...")
        torch.set_float32_matmul_precision('high')
        
        llm.forward = torch.compile(llm.forward, mode=args.compile_mode, fullgraph=True)
        if ssm is not None:
            ssm.forward = torch.compile(ssm.forward, mode=args.compile_mode, fullgraph=True)
    
    return model, tokenizer

def run_eval(llm_path, ssm_path, out_dir, args, dataset, log_dir, dtype=torch.float16, repeat=1):
    
    # Initialize
    model, tokenizer = load_model(llm_path, ssm_path, out_dir=out_dir, dtype=dtype, device="auto", args=args)

    # Warm up the model
    for i in trange(10, desc='Warming up'):
        input_message = f"Generate a long article about William Shakespeare."
        system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        warmup_input = [{"role": "system", "content": system_message},
                        {"role": "user", "content": input_message}]
        input_ids = tokenizer.apply_chat_template(warmup_input, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            model.generate(input_ids, temperature=args.temp, max_new_tokens=args.max_new_tokens, max_length=args.max_length, do_sample=args.do_sample)
    
    # Evaluate dataset
    avg_tput_list, avg_accept_rate_list = [], []
    for i in trange(repeat, desc="Repeat"):
        log_file = os.path.join(log_dir, f"{i}.jsonl")
        tput_list, accept_rate_list = [], []
        for idx, query in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating", leave=False):
            input_message = query.replace("[INST]", "").replace("[/INST]\n\nASSISTANT:", "")
            system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_message}
            ]
            input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                output_ids = model.generate(input_ids, temperature=args.temp, max_new_tokens=args.max_new_tokens, max_length=args.max_length, do_sample=args.do_sample)
            output = tokenizer.decode(output_ids[0][input_ids.shape[1]:])

            exp_log = {**model.exp_log, "query": input_message, "response": output}
            with open(log_file, 'a+') as f:
                json.dump(exp_log, f, indent=4)
                f.write("\n")

            tput_list.append(exp_log.get("tput", 0))
            if "sd" in args.mode:
                accept_rate_list.append(exp_log.get("avg_sampled", 0))
        
        print(f"Run {i+1}/{args.repeat}:")
        avg_tput = np.mean(tput_list)
        avg_accept_rate = np.mean(accept_rate_list) if accept_rate_list else 0
        print(f"\tThroughput: {avg_tput:.2f} tokens/sec")
        print(f"\tAcceptance rate: {avg_accept_rate:.2f} tokens/iter")
        
        avg_tput_list.append(avg_tput)
        avg_accept_rate_list.append(avg_accept_rate)

    print(f"Final Results:")
    tput_mean, tput_std = np.mean(avg_tput_list), np.std(avg_tput_list)
    accept_rate_mean, accept_rate_std = np.mean(avg_accept_rate_list), np.std(avg_accept_rate_list)
    print(f"\tThroughput: {tput_mean:.2f} ± {tput_std:.2f} tokens/sec")
    print(f"\tAcceptance rate: {accept_rate_mean:.2f} ± {accept_rate_std:.2f} tokens/iter")
    
    return tput_mean, accept_rate_mean    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-path", "-llm", required=True, help="Path to LLM weights.")
    parser.add_argument("--ssm-path", "-ssm", default="", help="Path to SSM weights.")
    parser.add_argument("--mode", default="naive", help="Model mode.")
    
    parser.add_argument(
        "--cache-impl",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic"
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default='eager',
        choices=["eager", 'reduce-overhead', 'max-autotune']
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Max number of draft iterations",
    )
    parser.add_argument(
        "--topk-len",
        type=int,
        default=10,
        help="Number of top draft nodes to keep on each draft iteration",
    )
    parser.add_argument(
        "--max-verify-tokens",
        type=int,
        default=60,
        help="Number of draft tokens to be verified at once.",
    )
    parser.add_argument(
        "--min-accept-prob",
        type=float,
        default=1e-2,
        help="All draft nodes should have probs higher than this value. (Currently not used)",
    )
    
    
    parser.add_argument("--out-dir", default="specdecodes/experiments/mt_bench/", help="Output directory.")
    parser.add_argument("--log-dir", default="specdecodes/experiments/result/", help="Experiment log directory.")
    parser.add_argument("--bench-name", default="mt_bench", help="Benchmark name.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="The maximum number of new generated tokens.")
    parser.add_argument("--max-length", type=int, default=1024, help="The maximum number of total tokens.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--repeat", type=int, default=3, help="Repeat evaluation.")
    parser.add_argument("--temp", type=float, default=0, help="Temperature for sampling.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
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
    
    avg_tput, avg_accept_rate = run_eval(
        args.llm_path, args.ssm_path, args.out_dir, args, 
        dataset, log_dir, 
        dtype=str_to_torch_dtype(args.dtype), repeat=args.repeat
    )