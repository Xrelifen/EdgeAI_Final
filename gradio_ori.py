from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import gradio as gr
import torch
import random
import threading
from collections import namedtuple
import time

stop_event = threading.Event()

def generate_configurations(target_model, device):
    layer_cnt = len(target_model.model.layers)

    # Offloading
    device_config = {}
    start = 12 # First 12 layers are kept on GPU
    end = layer_cnt
    for i in range(start, end):
        device_config[f"model.layers.{i}.self_attn.q_proj"] = 'cpu'
        device_config[f"model.layers.{i}.self_attn.k_proj"] = 'cpu'
        device_config[f"model.layers.{i}.self_attn.v_proj"] = 'cpu'
        device_config[f"model.layers.{i}.self_attn.o_proj"] = 'cpu'
        device_config[f"model.layers.{i}.mlp.gate_proj"] = 'cpu'
        device_config[f"model.layers.{i}.mlp.up_proj"] = 'cpu'
        device_config[f"model.layers.{i}.mlp.down_proj"] = 'cpu'

    # Set device map
    device_map = {}
    for name, _ in target_model.named_parameters():
        layer_name = ".".join(name.split(".")[:-1])
        if layer_name in device_config:
            device_map[layer_name] = 'cpu'
        else:
            device_map[layer_name] = device
    for name, _ in target_model.named_buffers():
        layer_name = ".".join(name.split(".")[:-1])
        device_map[layer_name] = device

    # Configs
    return device_map

def main():
    torch.manual_seed(0)
    random.seed(0)
    device = "cuda:0"

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    device_map = generate_configurations(model, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                baseline_chatbot = gr.Chatbot(type="messages")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    baseline_ttft = gr.Textbox(label="Time to First Token (sec)", value="0")
                    baseline_thrp = gr.Textbox(label="Tokens/sec", value="0")

        def update_input(prompt, baseline_chatbot: list):
            baseline_chatbot.append({"role": "user", "content": prompt})
            return "", baseline_chatbot

        def bots(baseline_chatbot: list):

            # Run baseline model
            baseline_chatbot.append({"role": "assistant", "content": ""})
            tokenized_chat = tokenizer.apply_chat_template(
                baseline_chatbot,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            baseline_kwargs = dict(
                inputs=tokenized_chat,
                streamer=streamer,
                max_new_tokens=2048,
            )
            start = time.perf_counter()
            last_time = start
            threading.Thread(target=model.generate, kwargs=baseline_kwargs).start()

            # Update model outputs
            b_ttft = 0
            b_thrp = 0
            ttft = False
            for new_text in streamer:
                if stop_event.is_set():
                    break
                current_token_time = time.perf_counter()
                if ttft == False:
                    b_ttft = current_token_time - start
                    ttft = True
                baseline_chatbot[-1]["content"] += new_text
                b_thrp = len(new_text) / (current_token_time - last_time)
                last_time = current_token_time
                yield baseline_chatbot, round(b_ttft, 5), round(b_thrp, 5)
            stop_event.clear()

        def stop_generate():
            stop_event.set()

        prompt_msg = gr.Textbox(label="Input")
        prompt_msg.submit(
            update_input,
            inputs=[prompt_msg, baseline_chatbot],
            outputs=[prompt_msg, baseline_chatbot]
        ).then(
            bots,
            inputs=[baseline_chatbot],
            outputs=[baseline_chatbot, baseline_ttft, baseline_thrp]
        )

        with gr.Row():
            stop = gr.Button(value="Stop Generating")
        stop.click(stop_generate)

        demo.launch()


if __name__ == "__main__":
    main()
