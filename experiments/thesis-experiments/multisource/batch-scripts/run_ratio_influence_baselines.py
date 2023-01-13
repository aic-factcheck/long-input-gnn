import os

BATCH_SCRIPT = "/home/hlavava1/afc/experiments/thesis-experiments/multisource/batch-scripts/00_baseline.sh"
DATASETS_PATH = "/home/hlavava1/afc/datasets/thesis-datasets/multisource/ratio-influence-40k"


def main():
    datasets_names = os.listdir(DATASETS_PATH)
    for dataset_name in datasets_names:
        if "40k-similar-10-1" in dataset_name:
            print("skipped", dataset_name)
            continue
        dataset_path = os.path.join(DATASETS_PATH, dataset_name)
        cmd = f"sbatch {BATCH_SCRIPT} {dataset_path}"
        os.system(cmd)


if __name__ == "__main__":
    main()
