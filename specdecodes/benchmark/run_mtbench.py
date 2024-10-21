import argparse
from tqdm import tqdm
import torch
import random
import os
import json
import numpy as np
from fastchat.llm_judge.common import load_questions
from fastchat.utils import str_to_torch_dtype

from transformers import AutoTokenizer, AutoModelForCausalLM
from ..models import HuggingFaceWrapper, NaiveWrapper, SDWrapper, ProfileSDWrapper, SharedKV_SDWrapper, SharedKV_ProfileSDWrapper, OffloadSDWrapper, ProfileOffloadSDWrapper
from ..models import SSM_Classic, SSM_Eagle, SSM_SharedKV, SSM_SX

# set random deterministic
torch.manual_seed(0)
random.seed(0)

def load_model(
    llm_path: str,
    ssm_path: str,
    mode: str,
    sd_method: str,
    layers: int,
    out_dir: str = None,
    dtype: torch.dtype = torch.float16,
    device: str = "auto",
    ):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
    
    # load LLM
    llm = AutoModelForCausalLM.from_pretrained(
        llm_path, 
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device
    )
    # check if ssm_path directory exists
    if os.path.exists(ssm_path):
        draft_config = deepcopy(llm.config)
        draft_config.num_hidden_layers = layers
    else:
        draft_config = None

    if mode == "naive":
        model = NaiveWrapper()
        
    elif mode == "hf":
        model = HuggingFaceWrapper()
    
    elif mode == "sd-classic":
        # model = SDWrapper()
        model = ProfileSDWrapper(out_dir=out_dir)
        
        # load SSM
        ssm = SSM_Classic.from_pretrained(
            ssm_path,
            config=draft_config,
            sampling_method=sd_method,
            eos_token_id=tokenizer.eos_token_id,
            torch_dtype=dtype,
        ).to(llm.model.layers[-1].self_attn.q_proj.weight.device)
        model.set_ssm(ssm)
        
    elif mode == "sd-eagle":
        # model = SDWrapper()
        model = ProfileSDWrapper(out_dir=out_dir)
        
        # load SSM
        ssm = SSM_Eagle.from_pretrained(
            ssm_path,
            config=draft_config,
            sampling_method=sd_method,
            eos_token_id=tokenizer.eos_token_id,
            torch_dtype=dtype,
        ).to(llm.model.layers[-1].self_attn.q_proj.weight.device)
        model.set_ssm(ssm)
        
    elif mode == "sd-sharedkv":
        # model = SharedKV_SDWrapper()
        model = SharedKV_ProfileSDWrapper(out_dir=out_dir)
        
        # load SSM
        ssm = SSM_SharedKV.from_pretrained(
            ssm_path,
            config=draft_config,
            sampling_method=sd_method,
            eos_token_id=tokenizer.eos_token_id,
            torch_dtype=dtype,
        ).to(llm.model.layers[-1].self_attn.q_proj.weight.device)
        model.set_ssm(ssm)
        
    else:
        raise ValueError("Invalid mode.")
    
    # set model
    model.set_tokenizer(tokenizer)
    model.set_llm(llm)
    model.eval()
    
    return model, tokenizer

def load_offload_model(
    llm_path: str,
    ssm_path: str,
    mode: str,
    sd_method: str,
    dtype: torch.dtype = torch.float16,
    device="cuda:0"    
):
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)

    if mode == "sd-offload":
        ssm = SSM_SX.from_pretrained(
            ssm_path,
            # config=draft_config,
            eos_token_id=tokenizer.eos_token_id,
            torch_dtype=dtype,
            sampling_method=sd_method,
        )
        ssm = ssm.to(device)

        # Load offload model
        # model = OffloadSDWrapper()
        model = ProfileOffloadSDWrapper()
        model.set_ssm(ssm)

    elif mode == "offload":
        model = OffloadWrapper()
        
    model.set_tokenizer(tokenizer)
    model.set_offload_llm(llm_path)
    model.eval()

    return model, tokenizer

def run_eval(
    llm_path,
    ssm_path,
    mode,
    sd_method,
    dataset,
    answer_file,
    max_new_tokens,
    temp=0.6,
    dtype=torch.float16,
    do_sample=False
):
    n_dataset = len(dataset)

    print("Loading model...")
    if "offload" in args.mode:
        model, tokenizer = load_offload_model(llm_path, ssm_path, mode, sd_method)
    else:
        raise NotImplementedError("Load model on gpu is not availabe yet")

    print("Warming up model...")

    # input message
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    input_message = "Hello."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_message},
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
    _  = model.generate(input_ids, temperature=temp, max_new_tokens=max_new_tokens, do_sample=do_sample)
    
    tput_list = []
    accept_rate_list = []

    print("Generating Response...")
    for i in tqdm(range(n_dataset), desc="Run Evaluation:"):
        # input message
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        input_message = dataset[i].replace("[INST]", "").replace("[/INST]\n\nASSISTANT:", "")
        
        # input_message = "Do you know what is Beyblade? What is the best strategy to build the strongest Beyblade?" # beyblade is the correct spelling

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_message},
        ]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
        output_ids = model.generate(input_ids, temperature=temp, max_new_tokens=max_new_tokens, do_sample=do_sample)
        output = model.tokenizer.decode(output_ids[0][input_ids.shape[1]:])
        
        exp_log = model.exp_log
        exp_log["query"] = input_message
        exp_log["response"] = output
        with open(answer_file, 'a+') as f:
            json.dump(exp_log, f, indent=4)
            f.write("\n")

        tput_list.append(exp_log["tput"])
        accept_rate_list.append(exp_log["accept_rate"])

    print(f"Average throughput: {np.mean(tput_list)} token / s")
    print(f"Average accept rate: {np.mean(accept_rate_list)}")
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-path",
        "-llm",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--ssm-path",
        "-ssm",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="naive",
        help="The mode of model generation.",
    )
    parser.add_argument(
        "--sd-method",
        type=str,
        default="greedy",
        help="The mode of model generation.",
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float16"
    )
    parser.add_argument(
        "--repeat",
        type=str,
        default=3,
        help="The number of times to repeat the evaluation.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Whether to do sampling. (Default is False)",
    )

    args = parser.parse_args()

    if args.bench_name == "mt_bench":
        file_path = "specdecodes/benchmark/data/mt_bench.json"
    else: 
        raise NotImplementedError("The dataset is currently not available yet")

    with open(file_path, "r") as f:
        dataset = json.load(f)
        dataset = [x[1] for x in dataset]
        random.shuffle(dataset)

    result_folder = "specdecodes/experiments/result"
    os.makedirs(result_folder, exist_ok=True)

    llm_name = args.llm_path.split("/")[-1]
    ssm_name = args.ssm_path.split("/")[-1]

    for i in range(args.repeat):
        answer_file = f"{result_folder}/{llm_name}-{ssm_name}-offload-{i}.jsonl"
        if os.path.exists(answer_file):
            os.remove(answer_file)

        run_eval(
            llm_path=args.llm_path,
            ssm_path=args.ssm_path,
            mode=args.mode,
            sd_method=args.sd_method,
            dataset=dataset,
            answer_file=answer_file,
            max_new_tokens=args.max_new_tokens,
            temp=args.temp,
            dtype=str_to_torch_dtype(args.dtype),
            do_sample=args.do_sample
        )