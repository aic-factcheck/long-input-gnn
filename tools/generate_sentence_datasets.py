# Tool to create the MultiSource datasets with different number of sentences
import os
import json


def generate_config_files():
    os.makedirs(
        "/home/hlavava1/afc/datasets/multisource/configs/fixed-sentences-count",
        exist_ok=True,
    )
    for n_random_sentences in range(1, 9):
        config = {
            "seed": 123,
            "splits": {
                "validation": [
                    {"n_examples": 1000, "n_sentences": 10, "n_random_sentences": 0},
                    {
                        "n_examples": 1000,
                        "n_sentences": 10,
                        "n_random_sentences": n_random_sentences,
                    },
                ],
                "test": [
                    {"n_examples": 1000, "n_sentences": 10, "n_random_sentences": 0},
                    {
                        "n_examples": 1000,
                        "n_sentences": 10,
                        "n_random_sentences": n_random_sentences,
                    },
                ],
                "train": [
                    {"n_examples": 10000, "n_sentences": 10, "n_random_sentences": 0},
                    {
                        "n_examples": 10000,
                        "n_sentences": 10,
                        "n_random_sentences": n_random_sentences,
                    },
                ],
            },
        }

        with open(
            f"/home/hlavava1/afc/datasets/multisource/configs/fixed-sentences-count/multisource-20k-similar-{n_random_sentences}-random-config.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(config, f)


def run_generators():
    config_filenames = os.listdir(
        "/home/hlavava1/afc/datasets/multisource/configs/fixed-sentences-count"
    )
    for config_filename in config_filenames:
        cmd = (
            "sbatch /home/hlavava1/afc/slurm/generate-multisource.batch "
            + "/home/hlavava1/afc/datasets/multisource/configs/fixed-sentences-count/"
            + config_filename
            + " /home/hlavava1/afc/datasets/multisource/fixed-sentences-count/"
            + config_filename.split(".")[0][:-7]
        )

        os.system(cmd)

        print(
            f"Started batch job for generating the dataset configured using {config_filename}"
        )
        print(f"\t{cmd}")


if __name__ == "__main__":
    generate_config_files()
    run_generators()
