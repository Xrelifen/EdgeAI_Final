import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TorchAoConfig
from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
)
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np
import os

import sglang as sgl
import argparse


#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################
def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    input_ids = test_enc.input_ids.to(device)

    nsamples = input_ids.numel() // model.seqlen
    nlls = []

    for i in tqdm(range(nsamples), desc="Evaluating PPL…"):
        batch = input_ids[:, (i * model.seqlen) : ((i + 1) * model.seqlen)]
        with torch.no_grad():
            outputs = model(batch)
            lm_logits = outputs.logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    return ppl.item()


def main(args):
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)

    max_new_tokens = 256  # Number of new tokens to generate
    device = "cuda:0"

    ### === TODO: Load your model (you may change this part) ===

    # model_name = "../dist/models/Llama-3.2-3B-Instruct"
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_name = "JKroller/llama3.2-3b-distill-to-1b"
    # model_name = "AMead10/Llama-3.2-3B-Instruct-AWQ"
    # model = pipeline(model_name, device=device)

    model = sgl.Engine(
        model_path=model_name,
        torchao_config="int8dq",
        kv_cache_dtype="auto",
        attention_backend="flashinfer",
        sampling_backend="pytorch",
        enable_torch_compile=False,
        mem_fraction_static=0.8,
    )
    sampling_params = {
        "max_new_tokens": 256,
    }
    #####################################

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # === (Optional) Uncomment the following lines if using the custom generate() function. ===
    # model.prefill_forward = model.forward
    # gen_config = GenerationConfig(max_new_tokens=max_new_tokens, random_seed=0)
    warmup_prompt = "Explain what AI is."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": warmup_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    # input_ids = inputs["input_ids"]
    # attention_mask = inputs["attention_mask"]

    # === (Optional) Set up StaticCache for manual KV cache management ===
    # from transformers import StaticCache
    # past_key_values = StaticCache(
    #     config=model.config,
    #     max_batch_size=1,
    #     max_cache_len=max_new_tokens + 16,
    #     device=model.device,
    #     dtype=torch.float16
    # )
    ####################################################################
    for i in tqdm(range(5), desc="Warm Up..."):
        #  === Default: use model.generate() for end-to-end warm-up ===
        # _ = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        # _ = model(warmup_prompt, generation_config=gen_config)
        _ = model.generate(prompt=warmup_prompt, sampling_params=sampling_params)

        # === (Optional) Use custom generate() if uncommented ===
        # generated = generate(model, input_ids, past_key_values, max_new_tokens)
        # past_key_values.reset()

    prompt = "How to learn a new language?"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # input_ids = inputs["input_ids"]
    # attention_mask = inputs["attention_mask"]
    tputs = []
    time_record = []
    # for _ in tqdm(range(10), desc="Test Inference"):
    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # === Default: Use model.generate() for end-to-end timing ===
        # generated = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        # generated = model(prompt, generation_config=gen_config)
        generated = model.generate(prompt=prompt, sampling_params=sampling_params)

        # === Optional: Use custom generate() if uncommented ===
        # generated = generate(model, input_ids, past_key_values, max_new_tokens)
        # past_key_values.reset()

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        tput = (
            generated["meta_info"]["completion_tokens"]
            / generated["meta_info"]["e2e_latency"]
        )
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)

    # response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    response = generated["text"]
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f"Prompt: {prompt}\nResponse: {response}\n")

    print(f"Time Record: {time_record}")
    print(f"Throughput Record: {tputs} toks/s\n")

    # ### Your final throughput result ###
    print(f"Throughput: {org_tput} toks/s")

    if hasattr(model, "shutdown"):
        model.shutdown()
    elif hasattr(model, "close"):
        model.close()
    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # ppl_model_name = "JKroller/llama3.2-3b-distill-to-1b"
    ppl_model_name =  model_name
    ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_name)
    quantization_config = TorchAoConfig("int8_dynamic_activation_int8_weight")
    ppl_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    if hasattr(ppl_model, "eval"):
        ppl_model.eval()

    ppl = evaluate_ppl(ppl_model, ppl_tokenizer, device)
    print(f"Perplexity (PPL): {ppl:.2f}")

    import csv

    with open("result.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "value"])
        writer.writerow([0, round(ppl, 2)])
        writer.writerow([1, round(org_tput, 1)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--quant_config", type=str, default="int8wo", help="Quantization configuration")
    parser.add_argument("--mem_fraction_static", type=float, default=0.8, help="Memory fraction for static cache")
    args = parser.parse_args()
    main(args)
