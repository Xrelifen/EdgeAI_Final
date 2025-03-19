from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm, trange
import os
import json
import numpy as np
import torch
import gc
import logging

def run_common_eval(generator, tokenizer, past_key_values, draft_past_key_values, args, dataset, log_dir, repeat):
    # Warm up the model
    is_profiling = generator.profiling
    generator.profiling = False
    for i in trange(args.warmup_iter, desc='Warming up'):
        input_message = "Write an essay about large language models."
        messages = [{"role": "user", "content": input_message}]
        tokenizer.use_default_system_prompt = True
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
            messages = [{"role": "user", "content": query}]
            tokenizer.use_default_system_prompt = True
            input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
            # logging.info(f"Check shape {input_ids.shape[1]}")
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                output_ids = generator.generate(input_ids, temperature=args.temperature, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_key_values, draft_past_key_values=draft_past_key_values)
            
            past_key_values.reset()
            if draft_past_key_values is not None:
                draft_past_key_values.reset()
            
            output_message = tokenizer.decode(output_ids[0][input_ids.shape[1]:])
            exp_log = {**generator.exp_log, "query": query, "response": output_message}
            with open(log_file, 'a+') as f:
                json.dump(exp_log, f, indent=4)
                f.write("\n")

            tput_list.append(exp_log.get("tput", 0))
            if exp_log.get("avg_sampled", None) is not None:
                accept_rate_list.append(exp_log.get("avg_sampled", 0))
                
            # gc.collect()
            # torch.cuda.empty_cache()
        
        print(f"Run {i+1}/{repeat}:")
        avg_tput = np.mean(tput_list)
        avg_accept_rate = np.mean(accept_rate_list) if accept_rate_list else 0
        print(f"\tThroughput: {avg_tput:.2f} tokens/sec")
        print(f"\tAcceptance rate: {avg_accept_rate:.2f} tokens/iter")
        
        # Write results to file
        with open(os.path.join(log_dir, "results.jsonl"), 'a+') as f:
            json.dump({i: {"tput": avg_tput, "accept_rate": avg_accept_rate}}, f, indent=4)
            f.write("\n")
        
        avg_tput_list.append(avg_tput)
        avg_accept_rate_list.append(avg_accept_rate)
        
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Final Results:")
    tput_mean, tput_std = np.mean(avg_tput_list), np.std(avg_tput_list)
    accept_rate_mean, accept_rate_std = np.mean(avg_accept_rate_list), np.std(avg_accept_rate_list)
    print(f"\tThroughput: {tput_mean:.2f} ± {tput_std:.2f} tokens/sec")
    print(f"\tAcceptance rate: {accept_rate_mean:.2f} ± {accept_rate_std:.2f} tokens/iter")
    
    return tput_mean, accept_rate_mean

def run_mtbench_eval(generator, tokenizer, past_key_values, draft_past_key_values, args, dataset, log_dir, repeat):
    # Warm up the model
    is_profiling = generator.profiling
    generator.profiling = False
    for i in trange(args.warmup_iter, desc='Warming up'):
        input_message = "Write an essay about large language models."
        messages = [{"role": "user", "content": input_message}]
        tokenizer.use_default_system_prompt = True
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
        for idx, turns in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating", leave=False):
            # org_len = 0
            exp_log = {}
            tmp_exp_log = {'total_sampled': 0, 'total_draft_time': 0, 'total_target_time': 0, 'total_verify_time': 0, 'n_iter': 0, 'n_tokens': 0, 'elapsed_time': 0}
            messages = []
            for tid, query in enumerate(turns):
                # print(f"Turn {tid+1}/{len(turns)} -> {query}"
                messages.append({"role": "user", "content": query})
                tokenizer.use_default_system_prompt = True
                input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
                # logging.info(f"Check shape {input_ids.shape[1]}")
                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    output_ids = generator.generate(input_ids, temperature=args.temperature, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_key_values, draft_past_key_values=draft_past_key_values)
                
                output_message = tokenizer.decode(output_ids[0][input_ids.shape[1]:])

                n_iter = generator.exp_log.get('n_iter', 0)
                n_tokens = generator.exp_log.get('n_tokens', 0)
                elapsed_time = generator.exp_log.get('elapsed_time', 0)
                
                tmp_exp_log['n_iter'] += n_iter
                tmp_exp_log['n_tokens'] += n_tokens
                tmp_exp_log['elapsed_time'] += elapsed_time
                
                tmp_exp_log['total_sampled'] += np.round(generator.exp_log.get('avg_sampled', 0) * n_iter, decimals=0)
                tmp_exp_log['total_draft_time'] += generator.exp_log.get('avg_draft_time', 0) * n_iter
                tmp_exp_log['total_target_time'] += generator.exp_log.get('avg_target_time', 0) * n_iter
                tmp_exp_log['total_verify_time'] += generator.exp_log.get('avg_verify_time', 0) * n_iter
                
                
                # org_len = output_ids.shape[1]
                # print(f'org_len: {org_len}')
                
                exp_log = {**exp_log, tid: {**generator.exp_log, f"query": query, f"response": output_message}}
                messages = []
                messages.append({"role": "system", "content": output_message})
            
            past_key_values.reset()
            if draft_past_key_values is not None:
                draft_past_key_values.reset()
            
            # output_message = tokenizer.decode(output_ids[0][input_ids.shape[1]:])
            exp_log = {
                **exp_log,
                "overall": {
                    "avg_draft_time": tmp_exp_log['total_draft_time'] / tmp_exp_log['n_iter'], 
                    "avg_target_time": tmp_exp_log['total_target_time'] / tmp_exp_log['n_iter'], 
                    "avg_verify_time": tmp_exp_log['total_verify_time'] / tmp_exp_log['n_iter'], 
                    "n_iter": tmp_exp_log['n_iter'], 
                    "n_tokens": tmp_exp_log['n_tokens'], 
                    "avg_sampled": tmp_exp_log['total_sampled'] / tmp_exp_log['n_iter'], 
                    "elapsed_time": tmp_exp_log['elapsed_time'],
                    "tput": tmp_exp_log['n_tokens'] / tmp_exp_log['elapsed_time']                    
                } 
            }
            
            with open(log_file, 'a+') as f:
                json.dump(exp_log, f, indent=4)
                f.write("\n")

            tput_list.append(generator.exp_log.get("tput", 0))
            if generator.exp_log.get("avg_sampled", None) is not None:
                accept_rate_list.append(generator.exp_log.get("avg_sampled", 0))
                
            # gc.collect()
            # torch.cuda.empty_cache()
        
        print(f"Run {i+1}/{repeat}:")
        avg_tput = np.mean(tput_list)
        avg_accept_rate = np.mean(accept_rate_list) if accept_rate_list else 0
        print(f"\tThroughput: {avg_tput:.2f} tokens/sec")
        print(f"\tAcceptance rate: {avg_accept_rate:.2f} tokens/iter")
        
        # Write results to file
        with open(os.path.join(log_dir, "results.jsonl"), 'a+') as f:
            json.dump({i: {"tput": avg_tput, "accept_rate": avg_accept_rate}}, f, indent=4)
            f.write("\n")
        
        avg_tput_list.append(avg_tput)
        avg_accept_rate_list.append(avg_accept_rate)
        
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Final Results:")
    tput_mean, tput_std = np.mean(avg_tput_list), np.std(avg_tput_list)
    accept_rate_mean, accept_rate_std = np.mean(avg_accept_rate_list), np.std(avg_accept_rate_list)
    print(f"\tThroughput: {tput_mean:.2f} ± {tput_std:.2f} tokens/sec")
    print(f"\tAcceptance rate: {accept_rate_mean:.2f} ± {accept_rate_std:.2f} tokens/iter")
    
    return tput_mean, accept_rate_mean