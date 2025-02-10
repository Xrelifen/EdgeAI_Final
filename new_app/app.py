import typer


def run_app(runner):
    app = typer.Typer()

    @app.command()
    def run_test():
        """
        Example subcommand for a test run.
        Usage:
            python custom.py run-test
        """
        pipeline, tokenizer = runner.build_pipeline()
        print(f"Running test with pipeline: {pipeline}")
        
    @app.command()
    def run_benchmark(bench_name: str = "mt_bench"):
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