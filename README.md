# <img src="assets/logo.png" alt="Medusa" width="100" align="left"><i>SpecDecodes:</i> A Library for Speculative Decoding

<br>

## 1. Fast Run & Test:
Simple run the following bash script to test the code:
```bash
bash run_test.sh
```

**c. Speculative Decoding with Target Model Offloading**
```bash
LOGLEVEL=INFO CUDA_VISIBLE_DEVICES=0 python run_test.py --max-new-tokens 256 --temp 1.0 --do-sample --seed 999 --mode sd-offload --sd-method greedy -llm meta-llama/Llama-3.1-8B-Instruct -ssm meta-llama/Llama-3.2-1B-Instruct
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

- New MTBench runner
```bash
CUDA_VISIBLE_DEVICES=0 python -m specdecodes.benchmark.run_mtbench --llm-path meta-llama/Llama-3.1-8B-Instruct --ssm-path meta-llama/Llama-3.2-1B-Instruct --mode sd-offload
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

- [ ] Acelerate model using torch Inductor

- [ ] Rewrite data_gen
  - data_gen produces wrong masking. Currently only old version works.
  - Rewrite data_gen with cleaner code.

- [ ] Handle multiple GPU inference correctly
  - May require copying llm's embed_token in each GPU for efficiency.
  - May require refactoring wrapper and ssm's code.