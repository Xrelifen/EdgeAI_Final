python result_ori.py --model_name "meta-llama/Llama-3.1-8B-Instruct"
python result_ori.py --model_name "meta-llama/Llama-3.2-1B-Instruct"
python sgl.py 
python sgl.py --quant_config "int8dq"