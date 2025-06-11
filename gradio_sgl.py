from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import gradio as gr
import torch
import random
import threading
from collections import namedtuple
import time
import sglang as sgl
from tqdm.auto import tqdm
import argparse
import requests
import json

stop_event = threading.Event()

def update_input(prompt, baseline_chatbot: list):
    baseline_chatbot.append({"role": "user", "content": prompt})
    return "", baseline_chatbot

def generate_stream(baseline_chatbot: list):
    stop_event.clear()

    prompt = baseline_chatbot[-1]["content"]

    url = f"http://localhost:{3000}/v1/chat/completions"

    data = {
        "model": "JKroller/llama3.2-3b-distill-to-1b",
        "messages": [{"role": "user", "content": f"{prompt}"}],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 256,
        },
        "stream": True
    }

    response = requests.post(url, json=data, stream=True)

    start_time = time.perf_counter()
    first_token_time = None
    full_output = ""
    ttft_reported = False
    last_time = start_time

    for chunk in response.iter_lines(decode_unicode=False):
        if stop_event.is_set():
            break
        chunk = chunk.decode("utf-8")
        if chunk and chunk.startswith("data:"):
            if chunk == "data: [DONE]":
                break
            data = json.loads(chunk[5:].strip("\n"))
            output = data["choices"][0]["delta"]["content"]
            if output != None:
                if first_token_time is None and output:
                    first_token_time = time.perf_counter()
                    ttft = round(first_token_time - start_time, 3)
                    ttft_reported = True

                full_output += output

                current_time = time.perf_counter()
                elapsed = current_time - last_time
                last_time = current_time
                throughput = round(len(output) / elapsed, 2) if elapsed > 0 else 0

                updated_chatbot = baseline_chatbot[:-1] + [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": full_output}
                ]

                # Yield TTFT only once immediately after first token
                yield updated_chatbot, str(ttft) if ttft_reported else "", str(throughput)

def stop_generate():
    stop_event.set()

def main():

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                baseline_chatbot = gr.Chatbot(type="messages")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    baseline_ttft = gr.Textbox(label="Time to First Token (sec)", value="0")
                    baseline_thrp = gr.Textbox(label="Tokens/sec", value="0")

        prompt_msg = gr.Textbox(label="Input")
        prompt_msg.submit(
            update_input,
            inputs=[prompt_msg, baseline_chatbot],
            outputs=[prompt_msg, baseline_chatbot]
        ).then(
            generate_stream,
            inputs=[baseline_chatbot],
            outputs=[baseline_chatbot, baseline_ttft, baseline_thrp]
        )

        with gr.Row():
            stop = gr.Button(value="Stop Generating")
        stop.click(stop_generate)

        demo.launch()


if __name__ == "__main__":
    main()
