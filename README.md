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

## Optimization Tips
### Quantization

### Kernel Optimization

### Speculative Decoding
- [EAGLE2](https://huggingface.co/JKroller/llama3.2-3b-eagle)
  - max-new-tokens: 256
    - speedup: 1.5x
  - max-new-tokens: 1024
    - speedup: 2.2x

## Performance