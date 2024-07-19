import torch
from eagle.model.ea_model import EagleModelLlama
from fastchat.model import get_conversation_template
import argparse
import time


def main(args):
    def warmup(model):
        conv = get_conversation_template(args.model_type)

        if args.model_type == "llama-2-chat":
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
        elif args.model_type == "mixtral":
            conv = get_conversation_template("llama-2-chat")
            conv.system_message = ''
            conv.sep2 = "</s>"
        conv.append_message(conv.roles[0], "Hello")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if args.model_type == "llama-2-chat":
            prompt += " "
        input_ids = model.tokenizer([prompt]).input_ids
        input_ids = torch.as_tensor(input_ids).cuda()
        # for output_ids in model.ea_generate(input_ids):
        #     ol=output_ids.shape[1]
        _ = model.eagle_generate(input_ids)

    print("Loading model...")
    eagle_conf_path = '/home/nctu/scott306lr/LREAGLE/eagle/eagle-llama2-7b'
    model = EagleModelLlama.from_pretrained(
        eagle_conf_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        load_draft_weight=True,#False
        total_tokens=60
    )
    print("Loaded.")
    # input("enter to exit...")
    # for n, m in model.named_parameters():
    #     print(f"{n}: {m.device}")
    # exit(0)


    # set model to eval mode, and warmup the model
    model.eval()
    # warmup(model)

    #test device
    # for n, m in model.named_parameters():
    #     print(f"{n}: {m.device}")
    # exit(0)

    # input message
    your_message="What's the best way to start learning a new language?"

    if args.model_type == "llama-2-chat":
        conv = get_conversation_template("llama-2-chat")  
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + " "
    elif args.model_type == "vicuna":
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    input_ids=model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()

    # generate response
    print("Generating response...")
    start_time = time.time()
    output_ids = model.eagle_generate(input_ids, temperature=args.temp, max_new_tokens=args.max_new_token)
    # for output_ids in model.eagle_generate_generator(input_ids, temperature=args.temp, max_new_tokens=args.max_new_token):
    #     print(output_ids, end='')
    end_time = time.time()
    
    output=model.tokenizer.decode(output_ids[0][input_ids.shape[1]:])
    # output=model.tokenizer.decode(output_ids[0][input_ids.shape[1]:])
    # output = model.tokenizer.decode(output_ids[0], skip_special_tokens=True, spaces_between_special_tokens=False,
    #                                       clean_up_tokenization_spaces=True, )

    if not args.no_print_message:
        print("\nPrompt:")
        print(prompt)
        print("\nModel response:")
        print(output)
        print("\n-----------------------------------")
        print("Input tokens:", len(input_ids[0]))
        print("Output tokens:", len(output_ids[0][input_ids.shape[1]:]))
    
    if not args.no_print_time:
        print("Time:", end_time - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="llama-2-chat",choices=["llama-2-chat","vicuna","mixtral"], help="llama-2-chat or vicuna, for chat template")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0, #0.5,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="The base model path.",
    )
    parser.add_argument(
        "--EAGLE-model-path",
        type=str,
        default="yuhuili/EAGLE-llama2-chat-7B",
        help="The EAGLE model path.",
    )
    parser.add_argument(
        "-nm",
        "--no-print-message",
        action="store_true",
        help="Print the message.",
    )
    parser.add_argument(
        "-nt",
        "--no-print-time",
        action="store_true",
        help="Record the time.",
    )
    args = parser.parse_args()
    
    main(args)