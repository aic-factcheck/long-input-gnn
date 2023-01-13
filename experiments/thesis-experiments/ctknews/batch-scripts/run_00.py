import os

BATCH_SCRIPT = "/home/hlavava1/afc/experiments/thesis-experiments/hyperpartisan/batch-scripts/00_baseline.sh"
DATASET_PATH = "/home/hlavava1/afc/datasets/thesis-datasets/hyperpartisan"
MODELS = ["bert-base-cased", "deepset/roberta-base-squad2", "deepset/xlm-roberta-base-squad2", "xlm-roberta-base"]


def main():
    for model_name in MODELS:
        cmd = f"sbatch {BATCH_SCRIPT} {DATASET_PATH} {model_name}"
        os.system(cmd)


if __name__ == "__main__":
    main()
