import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import trange
import logging
import os
import nvtx

import gemlite
from specdecodes.models.utils.cache_utils import create_kv_cache


def main(generator, tokenizer, args):
    # set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)

    # deterministic
    torch.manual_seed(args.seed)
    
    #Load GemLite cache
    gemlite.core.GEMLITE_TRITON_RESTRICT_M = True
    gemlite.core.GemLiteLinear.load_config('/tmp/gemlite_config.json')

    # kv-cache
    if args.cache_implementation == "static":
        if args.max_length is not None:
            max_cache_len = args.max_length + args.draft_params.max_sample_tokens
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
            

    # warm up
    if args.warmup_iter > 0:
        print("Warming up... It will take some time for the first few iterations to run.")
        with nvtx.annotate("Warming up"):
            generator.profiling = False
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
                        generator.generate(input_ids, temperature=args.temperature, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_key_values, draft_past_key_values=draft_past_key_values)
                
                past_key_values.reset()
                if draft_past_key_values is not None:
                    draft_past_key_values.reset()
                    
            generator.profiling = True
            
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
            output_ids = generator.generate(input_ids, temperature=args.temperature, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_key_values, draft_past_key_values=draft_past_key_values)
    end_event.record()
    
    # Ensure all CUDA kernels are done.
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    
    total_time_s = start_event.elapsed_time(end_event) / 1000.0
    output = generator.tokenizer.decode(output_ids[0][input_ids.shape[1]:])
    # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

    if args.print_message:
        print("\nPrompt:")
        print(prompt)
        print("\nModel response:")
        print(output)
        print("\n-----------------------------------")
        print("Input tokens:", len(input_ids[0]))
        print("Output tokens:", len(output_ids[0][input_ids.shape[1]:]))
    
    if args.print_time:
        print("Time:", total_time_s)