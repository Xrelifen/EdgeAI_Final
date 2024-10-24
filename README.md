# <img src="assets/logo.png" alt="Medusa" width="100" align="left"><i>SpecDecodes:</i> A Library for Speculative Decoding

<br>

## 1. To Run & Test Speculative Decoding:

**a. Classic Speculative Decoding:**
```bash
LOGLEVEL=DEBUG CUDA_VISIBLE_DEVICES=0 python run_test.py --max-new-tokens 256 --temp 1.0 --do-sample -nw --mode sd-classic --sd-method greedy --seed 999 -llm meta-llama/Llama-2-7b-chat-hf -ssm TinyLlama/TinyLlama_v1.1 
```

**b. Eagle-based Speculative Decoding:**
```bash
LOGLEVEL=DEBUG CUDA_VISIBLE_DEVICES=0 python run_test.py --max-new-tokens 256 --temp 1.0 --do-sample -nw --seed 999 --mode sd-eagle --sd-method greedy -llm meta-llama/Llama-2-7b-chat-hf -ssm <SSM directory> --layers 1
```

**c. Speculative Decoding with Target Model Offloading**
```bash
LOGLEVEL=INFO CUDA_VISIBLE_DEVICES=0 python run_test.py --max-new-tokens 256 --temp 1.0 --do-sample --seed 999 --mode sd-offload --sd-method greedy -llm meta-llama/Llama-3.1-8B-Instruct -ssm meta-llama/Llama-3.2-1B-Instruct
```

## 2. Run MT-Bench Benchmark:

- Slower case

```bash
CUDA_VISIBLE_DEVICES=0 python -m specdecodes.benchmark.llm_judge.gen_sd_answer --model-id llama7b --dtype float16 --mode sd-classic --sd-method greedy -llm meta-llama/Llama-2-7b-chat-hf -ssm TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out-dir specdecodes/experiments/mt-bench/b/greedy-d9k15/l1kl
```

- New MTBench runner
```bash
CUDA_VISIBLE_DEVICES=0 python -m specdecodes.benchmark.run_mtbench --llm-path meta-llama/Llama-3.1-8B-Instruct --ssm-path meta-llama/Llama-3.2-1B-Instruct --mode sd-offload
```


## 3. To Train Eagle-Based SSM:

### 1. Generate Dataset

```bash
python -m train.data_gen.allocation --gpu_index 0 --outdir <dataset save location>
```

### 2. Train SSM

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29500 --num_processes=4 --mixed_precision=bf16 -m specdecodes.train.main --datadir <dataset location> --data-ratio 1 --savedir <save location> --wandb 
```

##