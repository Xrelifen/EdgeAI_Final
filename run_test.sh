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
SSM_PATH=~/checkpoints/eagle
SEED=9991
DO_WARMUP=True
MAX_NEW_TOKENS=500
DO_SAMPLE=False
TEMPERATURE=0

# mode options
MODE=naive #sd-eagle

# sd specific options
SD_METHOD=greedy
LAYERS=1

# args for run_test.py
args="--mode $MODE --sd-method $SD_METHOD -llm $LLM_PATH -ssm $SSM_PATH --layers $LAYERS --seed $SEED --max-new-tokens $MAX_NEW_TOKENS"
if [ $DO_WARMUP = False ]; then
  args="$args -nw"
fi
if [ $DO_SAMPLE = True ]; then
  args="$args --do-sample --temp $TEMPERATURE"
fi

# execute run_test.py
LOGLEVEL=$LOGLEVEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_test.py $args