#!/usr/bin/env bash
#
# Usage:
#   ./run.sh [-d <CUDA_DEVICE>] [-l <LOG_LEVEL>]
#
# Example:
#   ./run.sh -d 1 -l DEBUG
#
# Description:
#   This script runs the `run_test.py` file with various configuration options.
#   It takes optional flags:
#     -d : Sets the CUDA_VISIBLE_DEVICES (default: 0)
#     -l : Sets the LOGLEVEL (default: DEBUG)

###############################################################################
# Default values
###############################################################################
CUDA_VISIBLE_DEVICES=0
LOGLEVEL=DEBUG

###############################################################################
# Parse command-line arguments
###############################################################################
while getopts ":d:l:" opt; do
  case $opt in
    d)
      CUDA_VISIBLE_DEVICES=$OPTARG
      ;;
    l)
      LOGLEVEL=$OPTARG
      ;;
    \?)
      echo "Error: Invalid option -$OPTARG" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

###############################################################################
# Script configuration
###############################################################################
# Paths
LLM_PATH=meta-llama/Llama-2-7b-chat-hf
# LLM_PATH=meta-llama/Llama-3.1-8B-Instruct
# SSM_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
# SSM_PATH=~/checkpoints/eagle/official/EAGLE-llama2-chat-7B/

# Execution parameters
SEED=9991
WARMUP_ITER=10
DO_SAMPLE=False
TEMPERATURE=0

# Stopping criteria parameters (set only one)
#! Warning: Setting MAX_NEW_TOKENS may alter the maximum token length based on input_ids
# MAX_NEW_TOKENS=1024
MAX_LENGTH=1024

# drafting parameters
DRAFT_MAX_DEPTH=15
DRAFT_TOPK_LEN=4
DRAFT_MAX_VERIFY_TOKENS=128
DRAFT_MIN_ACCEPT_PROB=1e-2

# Mode can be one of: ["naive", "sd-classic", "sd-eagle"]
# MODE="naive"
# MODE="sd-classic"
# MODE="sd-awq"
# MODE="sd-eagle"
MODE="sd-share"

# KV-cache options: ["static", "dynamic"]
CACHE_IMPL="static"
# CACHE_IMPL="dynamic"

# Compile mode options: ["eager", "reduce-overhead", "max-autotune"]
# COMPILE_MODE="eager"
# COMPILE_MODE="reduce-overhead"
COMPILE_MODE="max-autotune"

# NVTX profiling
NVTX_PROFILING=True
# NVTX_PROFILING=False

###############################################################################
# Construct arguments for run_test.py
###############################################################################
args=(
  --mode "$MODE"
  --cache-impl "$CACHE_IMPL"
  -llm "$LLM_PATH"
  -ssm "$SSM_PATH"
  --seed "$SEED"
  --warmup-iter "$WARMUP_ITER"
  --compile-mode "$COMPILE_MODE"
  --max-depth "$DRAFT_MAX_DEPTH"
  --topk-len "$DRAFT_TOPK_LEN"
  --max-verify-tokens "$DRAFT_MAX_VERIFY_TOKENS"
  --min-accept-prob "$DRAFT_MIN_ACCEPT_PROB"
)

# Optional arguments
if [ ! -z "$MAX_NEW_TOKENS" ]; then
  args+=("--max-new-tokens" "$MAX_NEW_TOKENS")
fi

if [ ! -z "$MAX_LENGTH" ]; then
  args+=("--max-length" "$MAX_LENGTH")
fi

if [ "$DO_SAMPLE" = True ]; then
  args+=("--do-sample" "--temp" "$TEMPERATURE")
fi


  
  

###############################################################################
# Execute command
###############################################################################
if [ "$NVTX_PROFILING" = True ]; then
  # https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59
  echo "LOGLEVEL=$LOGLEVEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nsys profile python run_test.py ${args[*]}"
  LOGLEVEL="$LOGLEVEL" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --cudabacktrace=all --force-overwrite=true --python-sampling-frequency=1000 --python-sampling=true --cuda-memory-usage=true --gpuctxsw=true --python-backtrace -x true -o nsight_report \
    python run_test.py "${args[@]}"
else
  echo "LOGLEVEL=$LOGLEVEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_test.py ${args[*]} --logging"
  LOGLEVEL="$LOGLEVEL" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    python run_test.py "${args[@]}" --logging
fi