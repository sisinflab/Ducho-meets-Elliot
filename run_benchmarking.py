from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run training and evaluation.")
parser.add_argument('--setting', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
args = parser.parse_args()

# run_experiment(f"config_files/baby_{args.setting}.yml")
# run_experiment(f"config_files/music_{args.setting}.yml")
# run_experiment(f"config_files/office_{args.setting}.yml")
# run_experiment(f"config_files/{args.dataset}_{args.setting}.yml")
run_experiment(f"config_files/{args.dataset}_{args.setting}_{args.batch_size}.yml")

