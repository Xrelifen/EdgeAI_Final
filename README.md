# EdgeAI Final
## Run & Test
- Simple Run Origin
```
CUDA_VISIBLE_DEVICES=0 python result_ori.py
```
- Run Optimized Version
```
CUDA_VISIBLE_DEVICES=0 python result.py
```

## Original
- On RTX 2080 Ti
  - max-new-tokens: 256
    - Throughput: 41.08 toks/s
  - max-new-tokens: 1024
    - Throughput: 73.89 toks/s

## Optimization Tips
### Quantization

### Kernel Optimization
- [Flashinfer](https://github.com/flashinfer-ai/flashinfer/tree/main)
  - On RTX 2080 Ti
    - max-new-tokens: 256
      - Throughput: 55.05 toks/s
      - Speedup: 1.34x
    - max-new-tokens: 1024
      - Throughput: 166.59 toks/s
      - Speedup: 2.25x

### Speculative Decoding
- [EAGLE2](https://huggingface.co/JKroller/llama3.2-3b-eagle)
  - On RTX 2080 Ti
    - max-new-tokens: 256
      - Throughput: 49.01 toks/s
      - speedup: 1.19x
    - max-new-tokens: 1024
      - Throughput: 131.39 toks/s
      - speedup: 1.77x

## Performance