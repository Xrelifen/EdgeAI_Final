import torch
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy
import argparse
import time
import os
import logging
from torch.nn.attention import SDPBackend, sdpa_kernel

import specdecodes.models.llm.modeling_llama as modeling_llama
# from transformers.models.llama import modeling_llama
from specdecodes.models import (
    HuggingFaceWrapper, 
    NaiveWrapper, 
    ProfileNaiveWrapper, 
    SDWrapper, 
    ProfileSDWrapper, 
    OffloadSDWrapper, 
    ProfileOffloadSDWrapper, 
    OffloadWrapper, 
    ShareSDWrapper, 
    ProfileShareSDWrapper,
    OffloadShareSDWrapper,
    ProfileOffloadShareSDWrapper
)

from specdecodes.models import DraftParams, SSM_Classic, SSM_Eagle, SSM_QTIP, SSM_ShareSD

from hqq.core.quantize import *
from hf_sd.base import AutoHQQHFModel
# from hqq.models.hf.base import AutoHQQHFModel
from hqq.utils.patching import prepare_for_inference
import gemlite

import nvtx

def load_model(
    llm_path: str,
    ssm_path: str,
    dtype: torch.dtype = torch.float16,
    device: str = "auto",
    args: dict = {},
):

    nbits = 2
    group_size = 64
    dtype = torch.bfloat16 if nbits == 4 else torch.float16

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
    load_device = device if not args.offload else "cpu"
    
    # load LLM
    # llm = AutoModelForCausalLM.from_pretrained(
    # llm = modeling_llama.LlamaForCausalLM.from_pretrained(
    llm = modeling_llama.LlamaForCausalLM.from_pretrained(
        llm_path, 
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=load_device,
        _attn_implementation="sdpa",
    )


    ssm = None
    
    # check if ssm_path directory exists
    # if os.path.exists(ssm_path):
    #     draft_config = deepcopy(llm.config)
    #     draft_config.num_hidden_layers = 1
        
    # else:
    #     draft_config = None
    draft_config = None

    if args.mode == "naive":
        model = ProfileNaiveWrapper() if args.logging else NaiveWrapper()
        
    elif args.mode == "hf":
        model = HuggingFaceWrapper()
        
    elif args.mode.split("-")[0] == "sd":
        if args.mode == "sd-classic":
            # load SSM
            ssm = SSM_Classic.from_pretrained(
                ssm_path,
                config=draft_config,
                eos_token_id=tokenizer.eos_token_id,
                torch_dtype=dtype,
            ).to(llm.model.layers[-1].self_attn.q_proj.weight.device if not args.offload else device)
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
        elif args.mode == "sd-share":
            # Build the memo dictionary from the model's parameters (and optionally buffers)
            model_memo = {}
            for _, param in llm.named_parameters():
                model_memo[id(param)] = param
            for _, buf in llm.named_buffers():
                model_memo[id(buf)] = buf

            # Clone the model using the memo dictionary.
            qmodule = copy.deepcopy(llm, memo=model_memo)
            
            # quantize
            print("Quantizing model...")
            backend = "torchao_int4" if nbits == 4 else "gemlite"
            dtype = torch.bfloat16 if backend == "torchao_int4" else torch.float16

            # quantize only the center layers, extend base_quant_config with keys for each layer
            start = 0
            end = 31
            
            base_quant_config_a = BaseQuantizeConfig(nbits=2, group_size=64, axis=1)
            base_quant_config_b = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=1)
            quant_config = {}
            for i in range(start, end+1):
                quant_config[f"layers.{i}.self_attn.q_proj"] = base_quant_config_a
                quant_config[f"layers.{i}.self_attn.k_proj"] = base_quant_config_a
                quant_config[f"layers.{i}.self_attn.v_proj"] = base_quant_config_a
                quant_config[f"layers.{i}.self_attn.o_proj"] = base_quant_config_a
                quant_config[f"layers.{i}.mlp.gate_proj"] = base_quant_config_b
                quant_config[f"layers.{i}.mlp.up_proj"] = base_quant_config_b
                quant_config[f"layers.{i}.mlp.down_proj"] = base_quant_config_b

            AutoHQQHFModel.quantize_model(qmodule, quant_config=quant_config, compute_dtype=dtype, device=("cuda" if not args.offload else device), offload=args.offload)
            HQQLinear.set_backend(HQQBackend.PYTORCH)
            # HQQLinear.set_backend(HQQBackend.ATEN)

            #Load GemLite cache
            # gemlite.core.GEMLITE_TRITON_RESTRICT_M = True
            # gemlite.core.GemLiteLinear.load_config('/tmp/gemlite_config.json')

            prepare_for_inference(qmodule, backend=backend)
            # prepare_for_inference(qmodule)

            ssm = SSM_ShareSD.from_pretrained(qmodule, eos_token_id=tokenizer.eos_token_id, torch_dtype=dtype)
        else:
            raise ValueError("Invalid sd mode.")

        draft_params = DraftParams(
            max_depth=args.max_depth,
            topk_len=args.topk_len,
            min_accept_prob=args.min_accept_prob,
            max_verify_tokens=args.max_verify_tokens
        )
        print("Draft params:", draft_params)
        
        if args.offload and args.mode == "sd-share":
            model = ProfileOffloadShareSDWrapper(draft_params=draft_params, out_dir=None) if args.logging else OffloadShareSDWrapper(draft_params=draft_params)
        elif args.mode == "sd-share":
            model = ProfileShareSDWrapper(draft_params=draft_params, out_dir=None) if args.logging else ShareSDWrapper(draft_params=draft_params)
        elif args.offload and args.mode == "sd-classic":
            model = ProfileOffloadSDWrapper(draft_params=draft_params, out_dir=None) if args.logging else OffloadSDWrapper(draft_params=draft_params)
        else:
            model = ProfileSDWrapper(draft_params=draft_params, out_dir=None) if args.logging else SDWrapper(draft_params=draft_params)
        model.set_ssm(ssm)
        
    else:
        raise ValueError("Invalid mode.")

    # set model
    model.cache_implementation = args.cache_impl
    model.set_tokenizer(tokenizer)
    model.set_llm(llm)
    model.eval()
        
    if args.compile_mode != 'eager':
        print("Running with Torch Inductor...")
        torch.set_float32_matmul_precision('high')

        llm.forward = torch.compile(llm.forward, mode=args.compile_mode, dynamic=False, fullgraph=True)
        if ssm is not None:
            # ssm.prefill_forward = torch.compile(ssm.prefill_forward, mode=args.compile_mode, dynamic=False, fullgraph=True)
            ssm.forward = torch.compile(ssm.forward, mode=args.compile_mode, dynamic=False, fullgraph=True)

    return model, tokenizer

def main(args, dtype=torch.float16):
    torch._dynamo.config.inline_inbuilt_nn_modules=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)	

    # set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)

    # deterministic
    torch.manual_seed(args.seed)

    # load model
    print("Loading model...")
    if args.offload:
        model, tokenizer = load_model(args.llm_path, args.ssm_path, dtype=dtype, device="cuda:0", args=args)
    else:
        model, tokenizer = load_model(args.llm_path, args.ssm_path, dtype=dtype, device="auto", args=args)

    # warm up
    if args.warmup_iter > 0:
        print("Warming up... It will take some time for the first few iterations to run.")
        with nvtx.annotate("Warming up"):
            model.disable_logging = True
            for i in trange(args.warmup_iter, desc='Warming up'):
                # input message
                system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
                input_message = "Write an essay about large language models."
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_message},
                ]
                with nvtx.annotate("Warm up"):
                    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
                    with sdpa_kernel(backends=[SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                        _  = model.generate(input_ids, temperature=args.temp, max_new_tokens=args.max_new_tokens, max_length=args.max_length, do_sample=args.do_sample)
            model.disable_logging = False

    # input message
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    # input_message = "Extract the following information from the presented texts: The name of the book, the author, the main character, the year of publication. Output in the format of \"main character, book, author, year of publication\", one book per line.\na) In the realm of wizarding literature, a true standout is the work of J.K. Rowling. One of her books that left an indelible mark is 'Harry Potter and the Philosopher's Stone'. This iconic tale, published in 1997, tells the story of Harry, a young orphan who discovers his magical abilities on his 11th birthday. Soon, he finds himself at the Hogwarts School of Witchcraft and Wizardry, a place teeming with magic and adventure, located somewhere in Scotland.\nb) The magic of Middle-earth has entranced readers worldwide, thanks to the brilliance of J.R.R. Tolkien. In one of his seminal works, 'The Lord of the Rings: The Fellowship of the Ring', published in 1954, we meet Frodo Baggins, a brave hobbit tasked with the perilous quest of destroying the One Ring. The epic journey takes him from the peaceful Shire to the tumultuous regions of Middle-earth.\nc) In a galaxy far, far away, the imagination of L.E. Starlighter gives us 'The Prism Galaxy Chronicles: The Awakening of the Starcaster'. Published in 2028, the story is about Zylo, a humble spaceship mechanic, who unexpectedly discovers he's a Starcaster - a rare individual with the power to manipulate stardust. Set against the backdrop of an interstellar empire in turmoil, Zylo's destiny unfolds on numerous alien worlds, each with its unique cosmic charm."
    # input_message = "Do you know what is Beyblade? What is the best strategy to build the strongest Beyblade?"
    input_message = "What's the best way to start learning a new language?"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_message},
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
    prompt = tokenizer.decode(input_ids[0])
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
                  
    # generate response

    # torch.cuda.memory._record_memory_history()
    print("Generating response...")
    torch.cuda.cudart().cudaProfilerStart() # start profiling from here
    start_event.record()
    with nvtx.annotate("Generate"):
        with sdpa_kernel(backends=[SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            output_ids = model.generate(input_ids, temperature=args.temp, max_new_tokens=args.max_new_tokens, max_length=args.max_length, do_sample=args.do_sample)
    end_event.record()
    
    # Ensure all CUDA kernels are done
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    
    total_time_s = start_event.elapsed_time(end_event) / 1000.0
    output = model.tokenizer.decode(output_ids[0][input_ids.shape[1]:])
    # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

    if not args.no_print_message:
        print("\nPrompt:")
        print(prompt)
        print("\nModel response:")
        print(output)
        print("\n-----------------------------------")
        print("Input tokens:", len(input_ids[0]))
        print("Output tokens:", len(output_ids[0][input_ids.shape[1]:]))
    
    if not args.no_print_time:
        print("Time:", total_time_s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="The maximum number of total tokens.",
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
        default=64,
        help="Number of draft tokens to be verified at once.",
    )
    parser.add_argument(
        "--min-accept-prob",
        type=float,
        default=1e-2,
        help="All draft nodes should have probs higher than this value. (Currently not used)",
    )
    
    parser.add_argument(
        "--logging",
        action="store_true",
        help="Log output of the model.",
    )
    parser.add_argument(
        "--warmup-iter",
        type=int,
        default=20,
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
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Offload LLM."
    )
    parser.add_argument(
        "--max-mem",
        type=float,
        default=8.0,
        help="Set max mem usage for offload mode."
    )
    args = parser.parse_args()
    main(args)