import typer
from .pipelines.run_test import main as main_run_test

def run_app(builder):
    app = typer.Typer()

    @app.command()
    def run_test():
        """
        Example subcommand for a test run.
        Usage:
            python custom.py run-test
        """
        generator, tokenizer = builder.build_generator()
        main_run_test(generator, tokenizer, args=builder)
        
        
    @app.command()
    def run_benchmark(bench_name: str = "mt-bench"):
        """
        Example subcommand for benchmarking.
        Usage: 
            python custom.py run-benchmark --bench-name=mt_bench
        """
        print(f"Running benchmark '{bench_name}'")

    @app.command()
    def run_gradio():
        """
        Example subcommand for launching a Gradio demo.
        Usage:
            python custom.py run-gradio
        """
        print(f"Running Gradio")

    app()