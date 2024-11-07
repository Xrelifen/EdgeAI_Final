# read opts for CUDA_VISIBLE_DEVICES, LOGLEVEL, and other options
CUDA_VISIBLE_DEVICES=6
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

LLM_PATH=meta-llama/Llama-2-7b-chat-hf
SSM_PATH=~/checkpoints/shrinkeagle/pretrain/l1ce/model_10
# SSM_PATH=~/checkpoints/shrinkeagle/linear/new/l1ce/model_20
# SSM_PATH=~/checkpoints/shrinkeagle/linear/new/ignore-last/pretrain/l1ce/model_20
# SSM_PATH=~/checkpoints/eagle/d1/l1ce/fp32/model_20

MODE=sd-shrink-eagle
# MODE=sd-eagle
SD_METHOD=greedy
LAYERS=1
SEED=9991

LOGLEVEL=$LOGLEVEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_test.py -nw --max-new-tokens 180 --temp 1.0 --do-sample --mode $MODE --sd-method $SD_METHOD -llm $LLM_PATH -ssm $SSM_PATH --layers $LAYERS --seed $SEED