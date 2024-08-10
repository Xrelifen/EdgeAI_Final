import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache

import numpy as np
import matplotlib.pyplot as plt

import time
import logging

from ..models.ssm.eagle import DraftModel


logging.getLogger().setLevel(logging.INFO)


# allocating 40MB to match L2 cache size on A100
x = torch.empty(int(40 * (1024 ** 2)), dtype=torch.int8, device='cuda')
def flush_cache():
    x.zero_()


def load_model(llm_path, dtype=torch.float16):
    config = AutoConfig.from_pretrained(llm_path)
    draft_config = config
    draft_config.num_hidden_layers = 1
    draft_config._attn_implementation = "sdpa"
    
    # create new head and embed_tokens
    lm_head = nn.Linear(draft_config.hidden_size, draft_config.vocab_size, bias=False).to(dtype=dtype, device="cuda")
    embed_tokens = nn.Embedding(draft_config.vocab_size, draft_config.hidden_size, draft_config.pad_token_id).to(dtype=dtype, device="cuda")
    model = DraftModel(draft_config).to(dtype=dtype, device="cuda")
    
    return model, lm_head, embed_tokens


def prepare_data(config, batch_size, prev_tokens, new_tokens, dtype=torch.float16, device="cuda"):
    head_dim = config.hidden_size // config.num_attention_heads
    past_key_values = DynamicCache()
    for i in range(0, config.num_hidden_layers):
        cache_k = torch.randn(batch_size, config.num_attention_heads, prev_tokens, head_dim, dtype=dtype, device=device)
        cache_v = torch.randn(batch_size, config.num_attention_heads, prev_tokens, head_dim, dtype=dtype, device=device)
        past_key_values.update(cache_k, cache_v, i)
    
    hidden_states = torch.randn(batch_size, new_tokens, config.hidden_size, dtype=dtype, device=device)
    tokens = torch.randint(100, (batch_size, new_tokens), device=device)
    return past_key_values, hidden_states, tokens


@torch.no_grad() # Time per output token
def benchmark_tpot(model, lm_head, embed_tokens, input_data, repetitions=100):
    # Get the number of previous tokens
    past_key_values, hidden_states, tokens = input_data
    prev_tokens = past_key_values.get_seq_length()
    
    # Warmup steps
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(repetitions):
            output = model(hidden_states, tokens, embed_tokens, past_key_values=past_key_values)
            _ = lm_head(output.last_hidden_state)
            past_key_values.crop(prev_tokens)
    torch.cuda.current_stream().wait_stream(s)
    
    # Capture CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # _ = model(tokens, past_key_values=past_key_values)
        output = model(hidden_states, tokens, embed_tokens, past_key_values=past_key_values)
        _ = lm_head(output.last_hidden_state)
    # actually not required to crop past_key_values, since cudagraph will replay and read and write to the same memory locations
    # past_key_values.crop(prev_tokens)
    
    # Start and end events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    
    # Run the benchmark
    for i in range(repetitions):
        flush_cache()
        start_events[i].record()
        graph.replay()
        end_events[i].record()
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    latency = sum(times) / len(times)
    
    return latency


def main():
    # Set seed
    torch.manual_seed(0)
    
    # Parameters
    llm_path = "meta-llama/Llama-2-7b-chat-hf"
    dtype = torch.float16
    device = "cuda"
    
    # Parameters
    batch_size = 1
    repetitions = 10#100
    prev_tokens = 512#4096#2048#1024
    max_new_tokens = 512
    latencies = []
    
    # Load model and prepare data
    model, lm_head, embed_tokens = load_model(llm_path, dtype=dtype)
    for i in range(1, max_new_tokens+1):
        input_data = prepare_data(model.config, batch_size, prev_tokens, i, dtype=dtype, device=device) 
        latency = benchmark_tpot(model, lm_head, embed_tokens, input_data, repetitions=repetitions)
        
        logging.info(f"Finished. \nprevious_tokens: {prev_tokens} \nnew_tokens: {i} \nlatency: {latency:.2f} milliseconds")
        latencies.append(latency)
    
    # convert to numpy array, plot and save
    latencies = np.array(latencies)

    # save latencies
    np.save(f"ssm_prev_{prev_tokens}.npy", latencies)


if __name__ == "__main__":
    main()