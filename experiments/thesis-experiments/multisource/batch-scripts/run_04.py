import os
import re

CONFIG = {
    "window_size": [2],
    "n_random_sentences": [6],
    "connectivity": ["full", "local_plus_global"]
}

BATCH_SCRIPT = "/home/hlavava1/afc/experiments/thesis-experiments/multisource/batch-scripts/04_ratio_influence.sh"
DATASETS_PATH = "/home/hlavava1/afc/datasets/thesis-datasets/multisource/ratio-influence-40k"


def main():
    datasets_names = os.listdir(DATASETS_PATH)
    n_random_sentences = [
        int(re.search(r"10-(\d)-random", x).group(1))
        for x in datasets_names
    ]

    datasets_names = [
        x for x, y in zip(datasets_names, n_random_sentences)
        if y in CONFIG["n_random_sentences"]
    ]

    for dataset_name in datasets_names:
        for connectivity in CONFIG["connectivity"]:
            for window_size in CONFIG["window_size"]:
                dataset_path = os.path.join(DATASETS_PATH, dataset_name)
                cmd = f"sbatch {BATCH_SCRIPT} {dataset_path} {window_size} {connectivity}"
                os.system(cmd)


if __name__ == "__main__":
    main()
