# <img src="assets/logo.png" alt="Medusa" width="100" align="left"><i>SpecDecodes:</i> A Library for Speculative Decoding

<br>

## 1. Fast Run & Test:
Simple run the following bash script to test the code:
```bash
bash run.sh run.share_sd run-test
```

## 2. Run mt_bench Benchmark:

a. Naive LLM Decoding:
```bash
bash run.sh run.naive run-benchmark --bench-name=mt_bench
```

b. Classic Speculative Decoding:
```bash
bash run.sh run.classic_sd run-benchmark --bench-name=mt_bench
```

c. Eagle-based Speculative Decoding:
```bash
bash run.sh run.eagle_sd run-benchmark --bench-name=mt_bench
```

4. ShareSD-based Speculative Decoding:
```bash
bash run.sh run.share_sd run-benchmark --bench-name=mt_bench
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
- [ ] Use different system prompt according to model type in benchmarking
  - Currently, the system prompt is hard-coded in the benchmarking script.
  - Need to change the system prompt according to the model type.
- [ ] Allow speculative decoding to run using multiple GPUs
  - May require copying llm's embed_token in each GPU for efficiency.
  - May require refactoring wrapper and ssm's code.
- [ ] Support multiple batch size for inference
  - Currently only support batch size 1

Training
- [ ] Rewrite data_gen
  - data_gen produces wrong masking. Currently only old version works.
  - Rewrite data_gen with cleaner code.
