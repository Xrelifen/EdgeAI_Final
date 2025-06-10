# 💻 Environment Setup for `sglang`

This guide outlines the steps to set up the environment for using [`sglang`](https://github.com/sgl-project/sglang) with all necessary dependencies, including CUDA 12.2 support.  
**Note:** This setup is tested on an environment using an **NVIDIA RTX4060 GPU**.

---

## ✅ Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed
- An environment named `sgl` (or create one via `conda create -n sgl python=3.11`)
- NVIDIA RTX4060 GPU with appropriate CUDA drivers installed

---

## 📦 Installation Steps

Follow the steps below **in order** to ensure a successful setup:

```bash
# 1. Upgrade pip
pip install --upgrade pip

# 2. Install uv package manager
pip install uv

# 3. Install sglang with all dependencies
uv pip install "sglang[all]>=0.4.6.post5"

# 4. Set CPATH for proper header file locations
export CPATH=/home/{$USER}/miniconda3/envs/sgl/targets/x86_64-linux/include:$CPATH

# 5. Disable HF Transfer (optional but recommended)
export HF_HUB_ENABLE_HF_TRANSFER="false"

# 6. Install GCC and G++ via conda
conda install -c conda-forge gcc_linux-64 gxx_linux-64

# 7. Set environment variables for the correct compiler
export CC=/home/{$USER}/miniconda3/envs/sgl/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home/{$USER}/miniconda3/envs/sgl/bin/x86_64-conda-linux-gnu-g++

# 8. Install CUDA NVCC (v12.2) via conda
conda install -c conda-forge cuda-nvcc_linux-64=12.2

# 9. Install accelerate and torchao
pip install accelerate
pip install torchao>=0.9.0

```

## 🚀 Runs
```bash
python sgl.py
```

## Result
| Model                             | Framework  | Offloading | Quantization   | PPL   | Tput (toks/s) | Speedup     |
|-----------------------------------|------------|------------|----------------|-------|---------------|-------------|
| meta-llama/Llama-3.1-8B-Instruct  | SGLang     | ✅         | None           | 7.21  | 1.23          | 1.00x       |
| meta-llama/Llama-3.1-8B-Instruct  | HuggingFace| ❌         | bnb-int8       | 7.29  | 3.90          | 3.17x       |
| meta-llama/Llama-3.1-8B-Instruct  | HuggingFace| ❌         | bnb-int4       | 7.62  | 21.34         | 17.35x      |
| meta-llama/Llama-3.2-1B-Instruct  | SGLang     | ❌         | None           | 13.16 | 74.89         | 60.89x      |
| JKroller/llama3.2-3b-distill-to-1b| SGLang     | ❌         | None           | 11.20 | 90.55         | 73.62x      |
| JKroller/llama3.2-3b-distill-to-1b| SGLang     | ❌         | torchao-int8dq | 11.33 | 125.23        | **101.00x** |




