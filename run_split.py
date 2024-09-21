import os
import shutil
from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['baby', 'office', 'music', 'toys', 'beauty'], help="Dataset name.", required=True)
args = parser.parse_args()

if not (os.path.exists(f'./data/{args.dataset}/train.tsv') and os.path.exists(f'./data/{args.dataset}/val.tsv') and os.path.exists(f'./data/{args.dataset}/test.tsv')):
    run_experiment(f"config_files/split_{args.dataset}.yml")
    shutil.move(f'./data/{args.dataset}_splits/0/test.tsv', f'./data/{args.dataset}/test.tsv')
    shutil.move(f'./data/{args.dataset}_splits/0/0/train.tsv', f'./data/{args.dataset}/train.tsv')
    shutil.move(f'./data/{args.dataset}_splits/0/0/val.tsv', f'./data/{args.dataset}/val.tsv')
    shutil.rmtree(f'./data/{args.dataset}_splits/')
