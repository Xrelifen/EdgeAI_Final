import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os


def build_dataset(tokenizer, dataset_dir, start, end, num_proc, seed=42):  
    
    def preprocess_function(examples):
        new_examples = {
            "conversation": [],
            "input_ids": [],
            "loss_mask": []
        }
        
        # Iterate through each conversation example
        for i in range(len(examples['id'])):
            # System prompt definition
            sys_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            
            # replace role from gpt/human to user/assistant
            rename_roles = {"human": "user", "gpt": "assistant"}

            # Fetch the conversation and remove the first message if not from user
            source = examples['conversations'][i]
            if rename_roles[source[0]["from"]] != "user":
                source = source[1:]

            # Prepare chat history by combining user and system messages
            chat_history = []
            for sentence in source:
                role = rename_roles[sentence["from"]]
                message = " " + sentence["value"] if role == "assistant" else sentence["value"]
                chat_history.append({"role": role, "content": message})
            
            # Apply chat template (assuming this method integrates system prompt and formats the conversation), keep the original conversation
            conversation = tokenizer.apply_chat_template(chat_history, sys_prompt=sys_prompt, tokenize=False, add_generation_prompt=True)
            
            # Set padding token if not already set
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            # Tokenize the conversation
            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                truncation=True
            ).input_ids[0]

            # Initialize the loss mask
            loss_mask = torch.ones_like(input_ids)

            # Identify assistant responses and adjust the loss mask
            cur_len = 1
            loss_mask[:cur_len] = 0

            turns = conversation.split(' </s><s>')
            for i, turn in enumerate(turns):
                if not turn:
                    break

                turn_len = len(tokenizer(turn).input_ids)
                parts = turn.split("Assistant: ")
                if len(parts) != 2:
                    break

                instruction_len = len(tokenizer(parts[0]).input_ids) - 2  # Adjust for LLaMA's special tokens
                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len + 2

            # Zero out the loss mask beyond the conversation length
            loss_mask[cur_len:] = 0

            # Append the processed data to new examples
            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids.unsqueeze(0))
            new_examples["loss_mask"].append(loss_mask.unsqueeze(0))

        return new_examples

    ds = load_dataset(dataset_dir)['train']
    ds = ds.shuffle(seed=seed)
    ds1 = ds.select(range(start, end))
    original_columns1 = ds1.column_names
    
    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )
    ds1.set_format(type="torch")
    return ds1


@torch.no_grad()
def model_forward(model, data):
    input_ids = data["input_ids"]
    outputs = model(input_ids.cuda(), past_key_values=DynamicCache(), output_hidden_states=True)
    hidden_state = outputs.hidden_states[-1]
    
    return {
        "input_ids": input_ids.cpu().squeeze(0),
        "hidden_state": hidden_state.cpu().squeeze(0),
        "loss_mask": data["loss_mask"].cpu().squeeze(0),
    }


def writedata(name, data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    print("dtype:",dtype)
    
    # Init model
    llm_path = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(llm_path, device_map="auto", attn_implementation="sdpa", torch_dtype=dtype)
    model.eval()
    
    # Load dataset
    dataset_dir = "Aeala/ShareGPT_Vicuna_unfiltered"
    ds = build_dataset(tokenizer, dataset_dir, args.start, args.end, args.num_proc, seed=42)
    print(ds)

    # Create output directory if not exists
    outdir = os.path.join(args.outdir, f"gpu{str(args.index)}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Output directory:", outdir)
    for id, data in tqdm(enumerate(ds)):
        if id % 100 == 0:
            print(id, end="\t")
        if id % 1000 == 0:
            print("")
            
        outdata = model_forward(model, data)
        writedata(outdir, outdata)


# Command line arguments
parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0, 1, 2])
parser.add_argument('--outdir', type=str, default='outdir0')
parser.add_argument('--dtype', type=str, default='fp16')
parser.add_argument('--num_proc', type=int, default=16)
args = parser.parse_args()

# Run main function
main(args)