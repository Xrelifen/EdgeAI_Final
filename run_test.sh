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
SSM_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
# SSM_PATH=~/checkpoints/eagle/official/EAGLE-llama2-chat-7B/

# Execution parameters
SEED=9991
DO_WARMUP=True
MAX_NEW_TOKENS=1024
DO_SAMPLE=False
TEMPERATURE=0

# drafting parameters
DRAFT_MAX_DEPTH=6
DRAFT_TOPK_LEN=10
DRAFT_MIN_ACCEPT_PROB=1e-2

# Mode can be one of: ["naive", "sd-classic", "sd-eagle"]
MODE="sd-classic"
# MODE="sd-eagle"

# KV-cache options: ["static", "dynamic"]
CACHE_IMPL="static"
# CACHE_IMPL="dynamic"

# Compile mode options: ["eager", "reduce-overhead", "max-autotune"]
# COMPILE_MODE="eager"
# COMPILE_MODE="reduce-overhead"
COMPILE_MODE="max-autotune"

# NVTX profiling
# NVTX_PROFILING=True
NVTX_PROFILING=False

###############################################################################
# Construct arguments for run_test.py
###############################################################################
args=(
  --mode "$MODE"
  --cache-impl "$CACHE_IMPL"
  -llm "$LLM_PATH"
  -ssm "$SSM_PATH"
  --seed "$SEED"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --compile-mode "$COMPILE_MODE"
  --max-depth "$DRAFT_MAX_DEPTH"
  --topk-len "$DRAFT_TOPK_LEN"
  --min-accept-prob "$DRAFT_MIN_ACCEPT_PROB"
)

# Optional arguments
if [ "$DO_WARMUP" = False ]; then
  args+=("-nw")
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
    nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsight_report -f true --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --cudabacktrace=true --osrt-threshold=10000 -x true \
    python run_test.py "${args[@]}"
else
  echo "LOGLEVEL=$LOGLEVEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_test.py ${args[*]} --logging"
  LOGLEVEL="$LOGLEVEL" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    python run_test.py "${args[@]}" --logging
fi
