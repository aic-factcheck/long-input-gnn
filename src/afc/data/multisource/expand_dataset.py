"""
Script used for expanding already created MultiSource dataset with examples
containing different permutations of already used sentence combinations.
"""
import os
import json
import argparse
import numpy as np

DATASET_FILES = ["train.jsonl", "validation.jsonl", "test.jsonl"]


def expand_dataset(args):
    assert (
        args.n_permutations > 0
    ), "The number of permutations to be added has to be greater than zero."

    for fn in DATASET_FILES:
        assert os.path.exists(
            os.path.join(args.dataset_path, fn)
        ), f"Couldn't find the {fn} file"

    random_state = np.random.RandomState(args.seed)

    for split_filename in DATASET_FILES:
        with open(
            os.path.join(args.dataset_path, split_filename), "r", encoding="utf-8"
        ) as f_in, open(
            os.path.join(args.output_dir, split_filename), "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                example = json.loads(line)

                # Store the original example
                json.dump(example, f_out, ensure_ascii=False)
                f_out.write("\n")

                n_sentences = len(example["sentences"])

                if args.train_only and "train" not in split_filename:
                    continue

                # Permutate the order of sentences in the example and store the "new" example
                for perm_idx in range(args.n_permutations):
                    permutation = random_state.permutation(n_sentences)
                    example["sentences"] = [
                        example["sentences"][idx] for idx in permutation
                    ]
                    example["ids"] = [example["ids"][idx] for idx in permutation]
                    json.dump(example, f_out, ensure_ascii=False)
                    f_out.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--n-permutations",
        type=int,
        required=True,
        help="The number of new permutations to be added per example",
    )
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--seed", default=123, type=int)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    expand_dataset(args)
