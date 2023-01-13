"""
Script for loading the HotpotQA dataset and taking only the subset of its yes/no examples.
"""

import datasets

SETTINGS = ["fullwiki", "distractor"]
# The settings differ only in the test splits, and since we make use of the
# train split only, it does not affect the transformed dataset.
# The restriction to training dataset is so that an 'oracle' approach
# can be used.


def _transform_example(example):
    example["answer"] = 1 if example["answer"] == "yes" else 0

    return example


def load_hotpotqa(setting="fullwiki", seed=42):
    assert setting in SETTINGS, "Invalid dataset settings selected"

    dataset = datasets.load_dataset("hotpot_qa", setting)
    dataset = dataset.filter(lambda x: x["answer"] in ["yes", "no"])

    # The split takes place here because the test split of the original dataset
    # does not contain any binary examples
    train_val_split = dataset["train"].train_test_split(test_size=0.2, seed=seed)
    val_test_split = train_val_split["test"].train_test_split(test_size=0.2, seed=seed)

    dataset["train"] = train_val_split["train"]
    dataset["validation"] = val_test_split["train"]
    dataset["test"] = val_test_split["test"]

    dataset = dataset.map(_transform_example)

    return dataset
