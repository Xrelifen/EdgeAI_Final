import typer
from .pipelines.run_test import main as main_run_test
from .pipelines.run_benchmark import main as main_run_benchmark
import torch

def run_app(builder):
    app = typer.Typer()

    @app.command()
    def run_test():
        """
        Example subcommand for a test run.
        Usage:
            python custom.py run-test
        """
        # torch.cuda.memory._record_memory_history()
        
        generator, tokenizer = builder.build_generator()
        main_run_test(generator, tokenizer, args=builder)
        
        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        # torch.cuda.memory._record_memory_history(enabled=None)
        
        
    @app.command()
    def run_benchmark(bench_name: str = "mt_bench"):
        """
        Example subcommand for benchmarking.
        Usage: 
            python custom.py run-benchmark --bench-name=mt_bench
        """
        print(f"Running benchmark '{bench_name}'")
        generator, tokenizer = builder.build_generator()
        main_run_benchmark(generator, tokenizer, args=builder, bench_name=bench_name)

    @app.command()
    def run_gradio():
        """
        Example subcommand for launching a Gradio demo.
        Usage:
            python custom.py run-gradio
        """
        print(f"Running Gradio")

    app()