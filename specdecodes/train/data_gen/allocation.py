import argparse
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--outdir', type=str, default='0')
parser.add_argument('--gpu_index', type=str, default='0,1,2,3')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=68000)
parser.add_argument('--dtype', type=str, default='fp16')
parser.add_argument('--suffix', type=str, default='mufp16')
args = parser.parse_args()

import os
from concurrent.futures import ThreadPoolExecutor


s = args.start
e = args.end - 1

# gpus=[[1, 2, 3]]
gpus=[[int(i) for i in args.gpu_index.split(',')]]
num_p = len(gpus)

outdir = os.path.join(args.outdir, 'sharegpt_{}_{}_{}'.format(s, e, args.suffix))

# delete the directory if it exists
if os.path.exists(outdir):
    print("Directory already exists. Exiting...")
    exit(0)
    # os.system('rm -r ' + outdir)


def split_range(start, end, n, over=False):
    length = end - start + 1  # Include the end
    base_interval = length // n
    additional = length % n  # Get the remainder of the division
    intervals = []
    previous = start

    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append((previous, previous + current_interval - 1))  # '-1' because the end is inclusive
        previous += current_interval

    return intervals


def run_command(cmd):
    os.system(cmd)


if not os.path.exists(outdir):
    os.makedirs(outdir)


data_a = split_range(s, e, num_p, over=True)
commands = []
for i in range(num_p):
    index = i
    start = data_a[i][0]
    end = data_a[i][1]
    # gpu_index_str = [str(i) for i in gpu_index]
    # gpu_index_str=','.join(gpu_index_str)
    gpu_index = gpus[i]
    gpu_index_str = ' '.join(map(str, gpu_index))
    # gpu_index_str='['+gpu_index_str+']'

    # combine directory with ge_data_all_vicuna.py
    parent_directory = os.path.dirname(os.path.realpath(__file__))
    exec_dir = os.path.join(parent_directory, 'ge_data_all_llama2chat_old.py')
    command = "python {} --start={} --end={} --index={} --gpu_index {} --outdir {} --dtype {}".format(exec_dir, start, end, index,
                                                                                                gpu_index_str, outdir, args.dtype)
    commands.append(command)
# run_command(commands[0])
# commands=commands[:1]
with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for command in commands:
        executor.submit(run_command, command)
        print(command)
