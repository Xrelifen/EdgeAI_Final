import torch
from tqdm import trange
import argparse
import os
import logging
from torch.nn.attention import SDPBackend, sdpa_kernel

import gemlite
from specdecodes.models import DraftParams, load_model, create_kv_cache
import nvtx


def main(args):
    # set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)

    # deterministic
    torch.manual_seed(args.seed)
    
    #Load GemLite cache
    gemlite.core.GEMLITE_TRITON_RESTRICT_M = True
    gemlite.core.GemLiteLinear.load_config('/tmp/gemlite_config.json')

    # load model
    print("Loading model...")
    # model, tokenizer = load_model(args.llm_path, args.ssm_path, dtype=dtype, device="auto", args=args)
    draft_params = DraftParams(
        max_depth=args.max_depth,
        topk_len=args.topk_len,
        max_verify_tokens=args.max_verify_tokens,
        min_accept_prob=args.min_accept_prob,
    )
    model, tokenizer = load_model(
        args.llm_path, args.ssm_path, args.mode,
        args.cache_impl, args.compile_mode,
        logging=args.logging,
        dtype=args.dtype, device=args.device,
        offload=args.offload,
        draft_params=draft_params,
        nbits=2,
        group_size=32,
        quant_range=(0, 31)
    )

    # kv-cache
    if args.max_length is not None and args.max_new_tokens is not None:
        raise ValueError(
            "Only one of max_length and max_new_tokens should be set."
        )
        
    if args.cache_impl == "static":
        if args.max_length is None and args.max_new_tokens is None:
            raise ValueError(
                "Either max_length and max_new_tokens should be set for 'static' kv-cache. Only 'dynamic' kv-cache is supported when length is unspecified."
            )
        elif args.max_length is not None:
            max_cache_len = args.max_length + draft_params.max_sample_tokens
              
        elif args.max_new_tokens is not None:
            print("'max_new_tokens' may cause max generation length dynamic. Please set 'max_length' to fix the generation length.")
            max_cache_len = model.llm.model.config.max_position_embeddings
            
        past_key_values = create_kv_cache(
            "static",
            max_cache_len=max_cache_len,
            max_batch_size=1,
            config=model.llm.model.config,
            device=model.llm.model.device,
            dtype=model.llm.model.dtype,
        )
        if args.mode == "sd-eagle" or args.mode == "sd-classic":
            ssm_past_key_values = create_kv_cache(
                "static",
                max_cache_len=max_cache_len,
                max_batch_size=1,
                config=model.ssm.model.config,
                device=model.ssm.model.device,
                dtype=model.ssm.model.dtype,
            )
        else:
            ssm_past_key_values = None
            
    else:
        past_key_values = create_kv_cache("dynamic")
        if args.mode == "sd-eagle" or args.mode == "sd-classic":
            ssm_past_key_values = create_kv_cache("dynamic")
        else:
            ssm_past_key_values = None
            

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
                    with sdpa_kernel(backends=[SDPBackend.MATH]):
                        model.generate(input_ids, temperature=args.temp, max_new_tokens=args.max_new_tokens, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_key_values, ssm_past_key_values=ssm_past_key_values)
                
                past_key_values.reset()
                if ssm_past_key_values is not None:
                    ssm_past_key_values.reset()
                    
            model.disable_logging = False
            
    gemlite.core.GemLiteLinear.cache_config('/tmp/gemlite_config.json')

    # input message
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    # input_message = "What's the best way to start learning a new language?"
    input_message = "Do you know what is Beyblade? What is the best strategy to build the strongest Beyblade?"
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
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = model.generate(input_ids, temperature=args.temp, max_new_tokens=args.max_new_tokens, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_key_values, ssm_past_key_values=ssm_past_key_values)
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
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Data type for the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for the model.",
    )
    
    def get_torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
        if not isinstance(dtype, torch.dtype):
            dtype = getattr(torch, dtype)
            assert isinstance(dtype, torch.dtype)
        return dtype
    
    args = parser.parse_args()
    args.dtype = get_torch_dtype(args.dtype)
    main(args)