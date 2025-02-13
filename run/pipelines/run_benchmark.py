from .benchmarks.run_mtbench import main as run_mtbench

def main(generator, tokenizer, args, bench_name):
    if bench_name == "mt_bench":
        run_mtbench(generator, tokenizer, args)
    else:
        raise NotImplementedError("Only 'mt_bench' dataset is supported.")