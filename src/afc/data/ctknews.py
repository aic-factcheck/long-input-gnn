import sqlite3
import pandas as pd
from datasets.dataset_dict import Dataset, DatasetDict
from rich.progress import Progress
from afc.utils.io import load_jsonl

DEFAULT_ARTICLES_DB_PATH = "/mnt/data/ctknews/factcheck/par5/interim/ctk_filtered.db"
LABELS = { "REFUTES": 0, "SUPPORTS": 1, "NOT ENOUGH INFO": 2 }


def get_evidence(example: dict, k: int = 10, articles_db_path: str = None):
    articles_db_path = (
        articles_db_path if articles_db_path else DEFAULT_ARTICLES_DB_PATH
    )
    k = (
        min(k, len(example["predicted_pages"]))
        if k
        else len(example["predicted_pages"])
    )

    evidence_ids = example["predicted_pages"][:k]

    with sqlite3.connect(articles_db_path) as con:
        df = pd.read_sql_query(
            "SELECT * FROM documents WHERE id IN ({})".format(
                "'" + "', '".join(evidence_ids) + "'"
            ),
            con,
        ).set_index("id")

        return df.loc[evidence_ids]["text"].to_list()


def dataset_list_to_dict(dataset: list, ignore_fields: list = None):
    keys = dataset[0].keys()
    if ignore_fields:
        keys = [ key for key in keys if key not in ignore_fields ]

    output = { key: [] for key in keys }
    for x in dataset:
        for key in keys:
            output[key].append(x[key])

    return output


def load_ctknews(split_paths: dict, k: int = 10, articles_db_path: str = None, display_progress: bool = True):
    # disable_caching()
    splits_data = {
        split_name: load_jsonl(path)
        for split_name, path in split_paths.items()
    }

    for split_name in splits_data.keys():
        with Progress() as progress:
            if display_progress:
                task = progress.add_task(f"Loading the evidence for {split_name}...", total=len(splits_data[split_name]))
            for example_idx in range(len(splits_data[split_name])):
                # map the label to an int
                example_label = splits_data[split_name][example_idx]["label"]
                splits_data[split_name][example_idx]["label"] = LABELS[example_label]
                splits_data[split_name][example_idx]["evidence"] = get_evidence(
                    splits_data[split_name][example_idx], k=k, articles_db_path=articles_db_path
                )
                if display_progress:
                    progress.update(task, advance=1)

    splits_data = {
        split_name: Dataset.from_dict(dataset_list_to_dict(dataset, ignore_fields=["id", "verifiable", "predicted_pages"]))
        for split_name, dataset in splits_data.items()
    }

    return DatasetDict(splits_data)
