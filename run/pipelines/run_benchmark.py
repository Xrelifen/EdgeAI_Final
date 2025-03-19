from tqdm import tqdm, trange
import os
import json
import numpy as np
import time
import torch
import random
import logging
import gc

from .benchmarks.utils.eval import run_common_eval, run_mtbench_eval
from .benchmarks.mtbench import load_mtbench_dataset
from .benchmarks.humaneval import load_humaneval_dataset
from .benchmarks.gsm8k import load_gsm8k_dataset
from .benchmarks.alpaca import load_alpaca_dataset
from .benchmarks.cnndm import load_cnndm_dataset
from .benchmarks.aime import load_aime_dataset
from .benchmarks.gpqa import load_gpqa_dataset
from .benchmarks.math500 import load_math500_dataset
from .benchmarks.livecodebench import load_livecodebench_dataset

DATASET_LOADER = {
    "mt-bench": load_mtbench_dataset,
    "human-eval": load_humaneval_dataset,
    "gsm8k": load_gsm8k_dataset,
    "alpaca": load_alpaca_dataset,
    "cnn-dm": load_cnndm_dataset,
    "aime": load_aime_dataset,
    "gpqa": load_gpqa_dataset,
    "math-500": load_math500_dataset,
    "livecodebench": load_livecodebench_dataset,
}

BENCHMARK_EVALUATORS = {
    "mt-bench": run_mtbench_eval,
    "human-eval": run_common_eval,
    "gsm8k": run_common_eval,
    "alpaca": run_common_eval,
    "cnn-dm": run_common_eval,
    "aime": run_common_eval,
    "gpqa": run_common_eval,
    "math-500": run_common_eval,
    "livecodebench": run_common_eval,
}

MAX_SAMPLES = 200

def main(generator, tokenizer, past_kv, draft_past_kv, args, bench_name):
    args.log_dir = os.path.join(args.log_dir, bench_name)
    
    # set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)

    # deterministic
    # torch.manual_seed(args.seed)
    # Fix random seed to 0 for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    

    dataset = DATASET_LOADER[bench_name]()
    random.shuffle(dataset)
    num_samples = min(len(dataset), MAX_SAMPLES)
    dataset = dataset[:num_samples]
    gc.collect()
    
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
    avg_tput, avg_accept_rate = BENCHMARK_EVALUATORS[bench_name](generator, tokenizer, past_kv, draft_past_kv, args, dataset, log_dir, repeat=3)
    
    # Write results to file
    with open(os.path.join(log_dir, "results.jsonl"), 'a+') as f:
        json.dump({"average": {"tput": avg_tput, "accept_rate": avg_accept_rate}}, f, indent=4)
        f.write("\n")
    
    logging.info(f'Peak Memory: {torch.cuda.max_memory_reserved("cuda:0")/(1024 ** 3)} GiB')