import os
from datasets import load_dataset, load_from_disk


def load_multisource(path: str, with_embedding: bool = False):
    """Loads the MultiSource dataset.

    Returns datasets.DatasetDict with train, validation, and test splits.
    It is expected that the directory contains the train.jsonl, validation.jsonl, and test.jsonl files.

    Args:
        path: The path to the directory where the dataset splits are located.
        with_embedding: Indicates whether the dataset already contains sentence embedding and has been
                        saved using `OrderedDict.save_to_disk`.
    """
    assert os.path.isdir(path), "The provided path does not lead to a directory!"

    if with_embedding:
        dataset = load_from_disk(path)
        return dataset

    dataset = load_dataset(
        "json",  # Using the datasets' loading scripts for jsonl files
        data_dir=path,
        data_files={
            "train": os.path.join(path, "train.jsonl"),
            "validation": os.path.join(path, "validation.jsonl"),
            "test": os.path.join(path, "test.jsonl"),
        },
    )

    return dataset
