# <img src="assets/logo.png" alt="Medusa" width="100" align="left"><i>SpecDecodes:</i> A Library for Speculative Decoding

<br>

# Share-SD Prototype
This branch is only the prototype of the Share-SD project. Will be merged into the main repo after the project is finished.
### Current Progress:
- [x] Quant llm linears, can switch between org/quant on verify/draft. (Slow if not compiled)
- [x] Fix Problem: torch.compile on shared-kv using same model slows down inference.
- [x] Fix Problem: torch.compile using hqq as draft model fails.
- [x] Allow skipping several layers to be quantized.
- [x] Fix Problem: Gemlite kernel doesn't work well with torch.compile.
- [ ] Fix Problem: SD doesn't obtain 100% accuracy when ssm & llm uses the same model with no sampling.
- [ ] Integrate offloading with ShareSD.


## 1. Fast Run & Test:
Simple run the following bash script to test the code:
```bash
bash run_test.sh
```

## 2. Run MT-Bench Benchmark:

a. Naive LLM Decoding:
```bash
LOGLEVEL=INFO CUDA_VISIBLE_DEVICES=1 python -m specdecodes.benchmark.run_mtbench --dtype float16 -llm meta-llama/Llama-2-7b-chat-hf --mode naive --do-sample --temp 1.0 --log-dir <log directory>
```

b. Classic Speculative Decoding:
```bash
LOGLEVEL=INFO CUDA_VISIBLE_DEVICES=1 python -m specdecodes.benchmark.run_mtbench --dtype float16 -llm meta-llama/Llama-2-7b-chat-hf -ssm <draft model directory> --mode sd-classic --out-dir <out directory> --log-dir <log directory>
```

c. Eagle-based Speculative Decoding:
```bash
LOGLEVEL=INFO CUDA_VISIBLE_DEVICES=1 python -m specdecodes.benchmark.run_mtbench --dtype float16 -llm meta-llama/Llama-2-7b-chat-hf -ssm <draft model directory> --mode sd-eagle --do-sample --temp 1.0 --out-dir <out directory> --log-dir <log directory>
```

## 3. To Train Eagle-Based SSM:

### 1. Generate Dataset

```bash
python -m specdecodes.train.data_gen.allocation --gpu_index 0 --outdir <dataset save location>
```

### 2. Train SSM

```bash
bash train.sh
```

## TODO

Inference
- [x] Accelerated the model using Torch Inductor, achieving an additional 2.1× speedup on the compiled LLM and 4.48× with the compiled LLM + SD!
- [ ] Allow speculative decoding to run using multiple GPUs
  - May require copying llm's embed_token in each GPU for efficiency.
  - May require refactoring wrapper and ssm's code.
- [ ] Support multiple batch size for inference
  - Currently only support batch size 1

Training
- [ ] Rewrite data_gen
  - data_gen produces wrong masking. Currently only old version works.
  - Rewrite data_gen with cleaner code.