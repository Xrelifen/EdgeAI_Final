import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import trange
import logging
import os
import nvtx


def main(generator, tokenizer, past_kv, draft_past_kv, args):
    # set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)

    # deterministic
    torch.manual_seed(args.seed)

    # warm up
    if args.warmup_iter > 0:
        print(
            "Warming up... It will take some time for the first few iterations to run."
        )
        with nvtx.annotate("Warming up"):
            is_profiling = generator.profiling
            generator.profiling = False
            for i in trange(args.warmup_iter, desc="Warming up"):
                input_message = "Write an essay about large language models."
                messages = [{"role": "user", "content": input_message}]
                tokenizer.use_default_system_prompt = True
                with nvtx.annotate("Warm up"):
                    input_ids = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(args.device)
                    with sdpa_kernel(backends=[SDPBackend.MATH]):
                        generator.generate(
                            input_ids,
                            temperature=args.temperature,
                            max_length=args.max_length,
                            do_sample=args.do_sample,
                            past_key_values=past_kv,
                            draft_past_key_values=draft_past_kv,
                        )

                past_kv.reset()
                if draft_past_kv is not None:
                    draft_past_kv.reset()
            generator.profiling = is_profiling

    # input message
    input_message = "How to learn a new language?"
    messages = [{"role": "user", "content": input_message}]
    tokenizer.use_default_system_prompt = True
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(args.device)
    prompt = tokenizer.decode(input_ids[0])

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # generate response
    print("Generating response...")
    torch.cuda.cudart().cudaProfilerStart()  # start profiling from here
    start_event.record()
    with nvtx.annotate("Generate"):
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = generator.generate(
                input_ids,
                temperature=args.temperature,
                max_length=args.max_length,
                do_sample=args.do_sample,
                past_key_values=past_kv,
                draft_past_key_values=draft_past_kv,
            )
    end_event.record()

    # Ensure all CUDA kernels are done.
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    total_time_s = start_event.elapsed_time(end_event) / 1000.0
    output = generator.tokenizer.decode(output_ids[0][input_ids.shape[1] :])

    if args.print_message:
        print("\nPrompt:")
        print(prompt)
        print("\nModel response:")
        print(output)
        print("\n-----------------------------------")
        print("Input tokens:", len(input_ids[0]))
        print("Output tokens:", len(output_ids[0][input_ids.shape[1] :]))

    if args.print_time:
        print("Time:", total_time_s)
