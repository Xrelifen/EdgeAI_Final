### sd ###
# LOGLEVEL=DEBUG CUDA_VISIBLE_DEVICES=0 python run_test.py \
#     --max-new-tokens 1024 --temp 1.0 --do-sample -nw \
#     --mode sd-classic --sd-method greedy --seed 999 \
#     -llm meta-llama/Llama-3.1-8B-Instruct \
#     -ssm meta-llama/Llama-3.2-1B-Instruct 

### sd + quant ###
# LOGLEVEL=DEBUG CUDA_VISIBLE_DEVICES=0 python run_test.py \
#     --max-new-tokens 1024 --temp 1.0 --do-sample -nw \
#     --mode sd-classic --sd-method greedy --seed 999 \
#     -llm ting0602/Llama-3.1-8B-Instruct-w4-g128-autoawq-marlin \
#     -ssm meta-llama/Llama-3.2-1B-Instruct 

### offload ###
# LOGLEVEL=INFO CUDA_VISIBLE_DEVICES=0 python run_test.py \
#     --max-new-tokens 256 --temp 0.6 --seed 999 \
#     --mode offload --sd-method greedy \
#     -llm meta-llama/Llama-3.1-8B-Instruct \
#     -ssm meta-llama/Llama-3.2-1B-Instruct

### offload + quant ###
# LOGLEVEL=INFO CUDA_VISIBLE_DEVICES=0 python run_test.py \
#     --max-new-tokens 256 --temp 0.6 --seed 999 \
#     --mode offload --sd-method greedy \
#     -llm ting0602/Llama-3.1-8B-Instruct-w4-g128-autoawq-marlin \
#     -ssm meta-llama/Llama-3.2-1B-Instruct

### offload + sd ###
# LOGLEVEL=INFO CUDA_VISIBLE_DEVICES=0 python run_test.py \
#     --max-new-tokens 256 --temp 0.6 --seed 999 \
#     --mode sd-offload --sd-method greedy \
#     -llm meta-llama/Llama-3.1-8B-Instruct \
#     -ssm meta-llama/Llama-3.2-1B-Instruct

### offload + sd + quant ###
# LOGLEVEL=INFO CUDA_VISIBLE_DEVICES=0 python run_test.py \
#     --max-new-tokens 256 --temp 0.6 --seed 999 \
#     --mode sd-offload --sd-method greedy \
#     -llm ting0602/Llama-3.1-8B-Instruct-w4-g128-autoawq-marlin \
#     -ssm meta-llama/Llama-3.2-1B-Instruct

### offload + sd + qtip ###
LOGLEVEL=INFO CUDA_VISIBLE_DEVICES=0 python run_test.py \
    --max-new-tokens 256 --temp 0.6 --seed 999 \
    --mode sd-offload --sd-method greedy \
    -llm meta-llama/Llama-2-7b-chat-hf \
    -ssm relaxml/Llama-2-7b-chat-QTIP-2Bit \
    --depth 15 \
    --use-static-tree-cache


# LOGLEVEL=INFO CUDA_VISIBLE_DEVICES=0 python run_test.py \
#     --max-new-tokens 256 --temp 0.6 --seed 999 \
#     --mode sd-offload --sd-method greedy \
#     -llm meta-llama/Llama-2-7b-chat-hf \
#     -ssm TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#     --depth 15 \
#     --use-static-tree-cache