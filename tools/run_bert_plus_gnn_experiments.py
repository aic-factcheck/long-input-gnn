# Runs Bert + GNN experiment for datasets contained within specified folder
import argparse
import os

ROOT_DIR = "/home/hlavava1/afc"


def run_experiments(bert_base: bool):
    datasets = os.listdir(f"{ROOT_DIR}/datasets/multisource/fixed-sentences-count")
    batch_script = (
        f"sbatch {ROOT_DIR}/slurm/multisource_train_bert_base_amd.batch "
        if bert_base
        else f"sbatch {ROOT_DIR}/slurm/multisource_train_bert_plus_gnn_amd.batch "
    )

    for dataset in datasets:
        cmd = f"{batch_script} {ROOT_DIR}/datasets/multisource/fixed-sentences-count/{dataset}"
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-base", action="store_true")
    args = parser.parse_args()

    run_experiments(args.bert_base)
