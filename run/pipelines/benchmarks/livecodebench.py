# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """

""".strip()

# LIVECODEBENCH
def load_livecodebench_dataset():

    dataset = load_dataset("livecodebench/code_generation")
    dataset = [entry['question_content'] for entry in dataset['test']]

    return dataset