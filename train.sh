# read opts for CUDA_VISIBLE_DEVICES, LOGLEVEL, and other options
CUDA_VISIBLE_DEVICES=0,1,2,3
LOGLEVEL=DEBUG
PORT=29500
while getopts ":d:l:" opt; do
  case $opt in
    d) CUDA_VISIBLE_DEVICES=$OPTARG
    ;;
    l) LOGLEVEL=$OPTARG
    ;;
    p) PORT=$OPTARG
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Data options
DATADIR=~/datasets/eagle/last1/sharegpt_0_6799_mufp16/gpu0/
# DATADIR=~/datasets/eagle/no-norm/sharegpt_0_67999_mufp16/gpu0/
DATA_RATIO=1

# Model options
SAVEDIR=~/checkpoints/eagle/sl1-ce/
TRAIN_MODEL=eagle
KEEP_EMBEDDINGS=False

# TRAIN_MODEL=custom
# LAYERS=2
# KEEP_EMBEDDINGS=True

# train options
PRECISION=bf16
EPCOHS=20
LR=1e-4
BATCH_SIZE=4

# wandb logging option
# WANDB=False
WANDB=True
WANDB_PROJECT=specdecodes

# select file to execute according to the TRAIN_MODEL
MAIN_FILE=specdecodes.train.main_eagle
if [ $TRAIN_MODEL = "eagle" ]; then
  MAIN_FILE=specdecodes.train.main_eagle
elif [ $TRAIN_MODEL = "shrinkeagle" ]; then
  MAIN_FILE=specdecodes.train.main_shrinkeagle
elif [ $TRAIN_MODEL = "custom" ]; then
  MAIN_FILE=specdecodes.train.main_custom
fi

# args for train code
NUM_PROCESSES=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
args="--datadir $DATADIR --savedir $SAVEDIR --data-ratio $DATA_RATIO --epochs $EPCOHS --lr $LR --bs $BATCH_SIZE"
if [ $KEEP_EMBEDDINGS = True ]; then
  args="$args --keep-embeddings"
fi
if [ $WANDB = True ]; then
  args="$args --wandb --wandb-project $WANDB_PROJECT"
fi

# execute train code
LOGLEVEL=$LOGLEVEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --main_process_port $PORT --num_processes=$NUM_PROCESSES --mixed_precision=$PRECISION -m $MAIN_FILE $args
