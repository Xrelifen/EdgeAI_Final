# read opts for CUDA_VISIBLE_DEVICES, LOGLEVEL, and other options
CUDA_VISIBLE_DEVICES=0
LOGLEVEL=DEBUG
while getopts ":d:l:" opt; do
  case $opt in
    d) CUDA_VISIBLE_DEVICES=$OPTARG
    ;;
    l) LOGLEVEL=$OPTARG
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Basic options
LLM_PATH=meta-llama/Llama-2-7b-chat-hf
# SSM_PATH=~/checkpoints/eagle/ce/model_5/
# SSM_PATH=~/checkpoints/eagle/sl1-ce/model_5/
SSM_PATH=~/checkpoints/eagle/sl1ce/model_20/
SEED=9991
DO_WARMUP=True
MAX_NEW_TOKENS=200
DO_SAMPLE=False
TEMPERATURE=0

# mode options
MODE=sd-eagle
# MODE=naive

# NVTX_PROFILING=True
NVTX_PROFILING=False

# args for run_test.py
args="--mode $MODE -llm $LLM_PATH -ssm $SSM_PATH --seed $SEED --max-new-tokens $MAX_NEW_TOKENS"
if [ $DO_WARMUP = False ]; then
  args="$args -nw"
fi
if [ $DO_SAMPLE = True ]; then
  args="$args --do-sample --temp $TEMPERATURE"
fi
if [ $NVTX_PROFILING = True ]; then
  # print the command to be executed
  echo LOGLEVEL=$LOGLEVEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nsys profile python run_test.py $args
  LOGLEVEL=$LOGLEVEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nsys profile python run_test.py $args
else
  # execute run_test.py
  # print the command to be executed
  echo LOGLEVEL=$LOGLEVEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_test.py $args --profile
  LOGLEVEL=$LOGLEVEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_test.py $args --profile
fi