# Tool to create the MultiSource datasets with different number of sentences
import os
import json


def generate_config_files():
    os.makedirs(
        "/home/hlavava1/afc/datasets/multisource/configs/different-similarity",
        exist_ok=True,
    )
    for drop_top_m in [0, 20, 40, 60, 80, 100]:
        config = {
            "seed": 123,
            "similar_sampling_config": {
                "k_hits": 200,
                "drop_top_m": drop_top_m,
                "use_top_n": 20,
            },
            "splits": {
                "validation": [
                    {"n_examples": 1000, "n_sentences": 10, "n_random_sentences": 0},
                    {"n_examples": 250, "n_sentences": 10, "n_random_sentences": 2},
                    {"n_examples": 250, "n_sentences": 10, "n_random_sentences": 4},
                    {"n_examples": 250, "n_sentences": 10, "n_random_sentences": 6},
                    {"n_examples": 250, "n_sentences": 10, "n_random_sentences": 8},
                ],
                "test": [
                    {"n_examples": 1000, "n_sentences": 10, "n_random_sentences": 0},
                    {"n_examples": 250, "n_sentences": 10, "n_random_sentences": 2},
                    {"n_examples": 250, "n_sentences": 10, "n_random_sentences": 4},
                    {"n_examples": 250, "n_sentences": 10, "n_random_sentences": 6},
                    {"n_examples": 250, "n_sentences": 10, "n_random_sentences": 8},
                ],
                "train": [
                    {"n_examples": 10000, "n_sentences": 10, "n_random_sentences": 0},
                    {"n_examples": 2500, "n_sentences": 10, "n_random_sentences": 2},
                    {"n_examples": 2500, "n_sentences": 10, "n_random_sentences": 4},
                    {"n_examples": 2500, "n_sentences": 10, "n_random_sentences": 6},
                    {"n_examples": 2500, "n_sentences": 10, "n_random_sentences": 8},
                ],
            },
        }

        with open(
            f"/home/hlavava1/afc/datasets/multisource/configs/different-similarity/multisource-20k-similar-{drop_top_m}-{drop_top_m + 19}-config.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(config, f)


def run_generators():
    config_filenames = os.listdir(
        "/home/hlavava1/afc/datasets/multisource/configs/different-similarity"
    )
    for config_filename in config_filenames:
        cmd = (
            "sbatch /home/hlavava1/afc/slurm/generate-multisource.batch "
            + "/home/hlavava1/afc/datasets/multisource/configs/different-similarity/"
            + config_filename
            + " /home/hlavava1/afc/datasets/multisource/different-similarity/"
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
