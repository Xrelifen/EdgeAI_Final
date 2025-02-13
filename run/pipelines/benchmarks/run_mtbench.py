import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm, trange
import time
import torch
import random
import os
import json
import numpy as np
import logging

import gemlite
from specdecodes.models.utils.cache_utils import create_kv_cache


def run_eval(generator, tokenizer, past_key_values, draft_past_key_values, args, dataset, log_dir, repeat):
    # Warm up the model
    is_profiling = generator.profiling
    generator.profiling = False
    for i in trange(args.warmup_iter, desc='Warming up'):
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        input_message = "Write an essay about large language models."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_message},
        ]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
        
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            generator.generate(input_ids, temperature=args.temperature, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_key_values, draft_past_key_values=draft_past_key_values)

        past_key_values.reset()
        if draft_past_key_values is not None:
            draft_past_key_values.reset()
    generator.profiling = is_profiling
    
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
            print("Length of input_ids: ", input_ids.shape[1])
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                output_ids = generator.generate(input_ids, temperature=args.temperature, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_key_values, draft_past_key_values=draft_past_key_values)
            
            past_key_values.reset()
            if draft_past_key_values is not None:
                draft_past_key_values.reset()
            
            output_message = tokenizer.decode(output_ids[0][input_ids.shape[1]:])
            exp_log = {**generator.exp_log, "query": input_message, "response": output_message}
            with open(log_file, 'a+') as f:
                json.dump(exp_log, f, indent=4)
                f.write("\n")

            tput_list.append(exp_log.get("tput", 0))
            if exp_log.get("avg_sampled", None) is not None:
                accept_rate_list.append(exp_log.get("avg_sampled", 0))
        
        print(f"Run {i+1}/{repeat}:")
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

def main(generator, tokenizer, args):
    # set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)

    # deterministic
    # torch.manual_seed(args.seed)
    # Fix random seed to 0 for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    
    #Load GemLite cache
    gemlite.core.GEMLITE_TRITON_RESTRICT_M = True
    gemlite.core.GemLiteLinear.load_config('/tmp/gemlite_config.json')
    
    # kv-cache
    if args.cache_implementation == "static":
        if args.max_length is not None:
            if getattr(generator, 'draft_model', None) is not None:
                # Additional sample tokens may cause KV-Cache tp exceed max_length
                max_cache_len = args.max_length + args.draft_params.max_sample_tokens
            else:
                max_cache_len = args.max_length
        else:
            raise ValueError("max_length should be set for static cache.")
        
        # Create static kv-cache
        past_key_values = create_kv_cache(
            "static",
            max_cache_len=max_cache_len,
            max_batch_size=1,
            config=generator.target_model.model.config,
            device=generator.target_model.model.device,
            dtype=generator.target_model.model.dtype,
        )
        # if generator.draft_model is not None:
        if getattr(generator, 'draft_model', None) is not None:
            draft_past_key_values = create_kv_cache(
                "static",
                max_cache_len=max_cache_len,
                max_batch_size=1,
                config=generator.draft_model.model.config,
                device=generator.draft_model.model.device,
                dtype=generator.draft_model.model.dtype,
            )
        else:
            draft_past_key_values = None
            
    else:
        # Create dynamic kv-cache
        past_key_values = create_kv_cache("dynamic")
        # if generator.draft_model is not None:
        if getattr(generator, 'draft_model', None) is not None:
            draft_past_key_values = create_kv_cache("dynamic")
        else:
            draft_past_key_values = None

    with open("run/pipelines/benchmarks/data/mt_bench.json") as f:
        dataset = [x[1] for x in json.load(f)]
        random.shuffle(dataset)
    
    # Handle output directories
    if args.out_dir is not None:
        os.system(f"rm -rf {args.out_dir}")
        print(f"Deleted old {args.out_dir}")
        os.makedirs(args.out_dir, exist_ok=True)
        
    log_dir = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}")
    print(f"Log directory: {log_dir}")
            
    # Evaluate
    avg_tput, avg_accept_rate = run_eval(generator, tokenizer, past_key_values, draft_past_key_values, args, dataset, log_dir, repeat=3)
    
    # Write results to file
    with open(os.path.join(log_dir, "results.json"), 'w') as f:
        json.dump({"tput": avg_tput, "accept_rate": avg_accept_rate}, f, indent=4)