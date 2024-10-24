import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from copy import deepcopy
import argparse
import time
import os
import logging

from specdecodes.models import (
    HuggingFaceWrapper, 
    NaiveWrapper, 
    SDWrapper, 
    ProfileSDWrapper, 
    OffloadSDWrapper, 
    SharedKV_SDWrapper, 
    SharedKV_ProfileSDWrapper,
    OffloadWrapper,
    ProfileOffloadSDWrapper
) 
from specdecodes.models import SSM_Classic, SSM_Eagle, SSM_SharedKV, SSM_SX, SSM_SQ

# LOGLEVEL=INFO CUDA_VISIBLE_DEVICES=0 python run_test.py --max-new-tokens 256 --temp 1.0 --do-sample --seed 999 --mode sq-offload --sd-method greedy -llm meta-llama/Llama-2-7b-chat-hf -ssm TinyLlama/TinyLlama-1.1B-Chat-v1.0
# LOGLEVEL=INFO CUDA_VISIBLE_DEVICES=0 python run_test.py --max-new-tokens 256 --temp 1.0 --do-sample --seed 999 --mode sq-offload --sd-method greedy -llm meta-llama/Llama-3.1-8B-Instruct -ssm meta-llama/Llama-3.2-1B-Instruct

def load_model(
    llm_path: str,
    ssm_path: str,
    mode: str,
    sd_method: str,
    layers: int,
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
        model = ProfileSDWrapper(out_dir=None)
        
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
        model = ProfileSDWrapper(out_dir=None)
        
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
        model = SharedKV_ProfileSDWrapper(out_dir=None)
        
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
        # model = ProfileOffloadSDWrapper(out_dir='specdecodes/experiments/profile_data/llama3')
        model = OffloadSDWrapper()
        model.set_ssm(ssm)

    elif mode == "sq-offload":
        ssm = SSM_SQ.from_pretrained(
            ssm_path,
            # config=draft_config,
            eos_token_id=tokenizer.eos_token_id,
            torch_dtype=dtype,
            sampling_method=sd_method,
        )
        grow_map = torch.load('../Sequoia/demo_tree_512_new.pt')
        ssm.load_spectree_arch(grow_map['branches'])
        
        ssm = ssm.to(device)
        # Load offload model
        model = OffloadSDWrapper()
        model.set_ssm(ssm)
    

    elif mode == "offload":
        model = OffloadWrapper()

    else:
        raise ValueError("Invalid mode.")

    model.set_tokenizer(tokenizer)
    model.set_offload_llm(llm_path)
    model.eval()

    return model, tokenizer

def main(args):
    
    # set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(format='%(levelname)s - %(message)s', level=LOGLEVEL)

    # deterministic
    torch.manual_seed(args.seed)

    # load model
    print("Loading model...")
    if "offload" in args.mode:
        model, tokenizer = load_offload_model(args.llm_path, args.ssm_path, args.mode, args.sd_method)
    else:
        model, tokenizer = load_model(args.llm_path, args.ssm_path, args.mode, args.sd_method, args.layers)

    # warm up
    if not args.no_warm_up:
        print("Warming up model...")

        # input message
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        input_message = "Hello."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_message},
        ]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
        _  = model.generate(input_ids, temperature=args.temp, max_new_tokens=args.max_new_tokens, max_length=args.max_length, do_sample=args.do_sample)

    # generate response
    print("Generating response...")

    # input message
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    input_message = "What's the best way to start learning a new language?"
    # input_message = "Do you know what is Beyblade? What is the best strategy to build the strongest Beyblade?" # beyblade is the correct spelling

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_message},
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
    prompt = tokenizer.decode(input_ids[0])
    
    start_time = time.time()
    output_ids = model.generate(input_ids, temperature=args.temp, max_new_tokens=args.max_new_tokens, max_length=args.max_length, do_sample=args.do_sample)
    end_time = time.time()

    for key, value in model.exp_log.items():
        print(f"{key}: {value}")
    
    output = model.tokenizer.decode(output_ids[0][input_ids.shape[1]:])

    if not args.no_print_message:
        print("\nPrompt:")
        print(prompt)
        print("\nModel response:")
        print(output)
        print("\n-----------------------------------")
        print("Input tokens:", len(input_ids[0]))
        print("Output tokens:", len(output_ids[0][input_ids.shape[1]:]))
    
    if not args.no_print_time:
        print("Time:", end_time - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.6,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--llm-path",
        "-llm",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="LLM model path.",
    )
    parser.add_argument(
        "--ssm-path",
        "-ssm",
        type=str,
        default="/share3/saves/scott306lr/weights/eagle_with_ln",
        help="SSM model path.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Whether to do sampling. (Default is False)",
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
        "--layers",
        type=int,
        default=1,
        help="The number of layers for SSM.",
    )
    parser.add_argument(
        "-nw",
        "--no-warm-up",
        action="store_true",
        help="Warm up the model.",
    )
    parser.add_argument(
        "-nm",
        "--no-print-message",
        action="store_true",
        help="Print the message.",
    )
    parser.add_argument(
        "-nt",
        "--no-print-time",
        action="store_true",
        help="Record the time.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()
    
    main(args)