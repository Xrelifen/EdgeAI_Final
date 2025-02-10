#!/usr/bin/env bash
#
# Usage:
#   ./run.sh <python script>
#
# Example:
#   ./run.sh eagle.py
#
# Description:
#   This script adds numerous flags before running selected python script.

CUDA_VISIBLE_DEVICES=0
LOGLEVEL=DEBUG

NVTX_PROFILING=True
# NVTX_PROFILING=False  

###############################################################################
# Execute command
###############################################################################
if [ "$NVTX_PROFILING" = True ]; then
  # https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59
  echo "LOGLEVEL=$LOGLEVEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nsys profile python -m $@"
  LOGLEVEL="$LOGLEVEL" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --cudabacktrace=all --force-overwrite=true --python-sampling-frequency=1000 --python-sampling=true --cuda-memory-usage=true --gpuctxsw=true --python-backtrace -x true -o nsight_report \
    python -m $@
else
  echo "LOGLEVEL=$LOGLEVEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m $@"
  LOGLEVEL="$LOGLEVEL" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    python -m $@
fi