import os
import sqlite3
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import datasets
from argparse import ArgumentParser
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import Dataset, DatasetDict, load_from_disk
from datasets.fingerprint import Hasher
from torch.utils.data import DataLoader
from rich.progress import track, Progress
from typing import Tuple

import afc
from afc.models import BERTPlusGNNClassifier, GNNClassifier
from afc.models.utils import evaluate_model

EXPERIMENT_NAME = "ctknews-model-hyperparams"


ARTICLES_DB_PATH = "/mnt/data/ctknews/factcheck/par5/interim/ctk_filtered.db"
RETRIEVAL_RESULTS_BASE_PATH = "/mnt/data/ctknews/factcheck/dataset/splits_concise_ctkId_s0_si0_t0.095_v0.12_source_77/models/anserini/predictions"

RETRIEVAL_RESULTS_PATHS = {
    "train": os.path.join(RETRIEVAL_RESULTS_BASE_PATH, "train_anserini_k500_0.6_0.5.jsonl"),
    "validation": os.path.join(RETRIEVAL_RESULTS_BASE_PATH, "validation_anserini_k500_0.6_0.5.jsonl"),
    "test": os.path.join(RETRIEVAL_RESULTS_BASE_PATH, "test_anserini_k500_0.6_0.5.jsonl")
}


def get_evidence(x, k=None):
    k = min(k, len(x["predicted_pages"])) if k else len(x["predicted_pages"])
    evidence_ids = x["predicted_pages"][:k]

    with sqlite3.connect(ARTICLES_DB_PATH) as con:
        df = pd.read_sql_query("SELECT * FROM documents WHERE id IN ({})".format(
            "'" + "', '".join(evidence_ids) + "'"
        ), con).set_index("id")

        return df.loc[evidence_ids]["text"].to_list()


def convert_list_to_dict(dataset: list, ignore_fields: list = None):
    keys = [ key for key in dataset[0].keys() if key not in ignore_fields ]
    output = { key: [] for key in keys }
    for x in dataset:
        for key in keys:
            output[key].append(x[key])

    return output


def load_ctknews(paths: dict, k: int = 10):
    LABELS = { "REFUTES": 0, "SUPPORTS": 1, "NOT ENOUGH INFO": 2 }

    splits = {
        name: afc.utils.io.load_jsonl(path)
        for name, path in paths.items()
    }

    for split_name in splits.keys():
        for example_idx in range(len(splits[split_name])):
            splits[split_name][example_idx]["label"] = LABELS[
                splits[split_name][example_idx]["label"]
            ]
            splits[split_name][example_idx]["evidence"] = get_evidence(
                splits[split_name][example_idx], k=k
            )

    splits = {
        k: convert_list_to_dict(v, ignore_fields=["id", "verifiable", "predicted_pages"])
        for k, v in splits.items()
    }

    splits = {
        k: datasets.Dataset.from_dict(v)
        for k, v in splits.items()
    }

    return datasets.DatasetDict(splits)


class CTKNewsDataModule(LightningDataModule):
    def __init__(
        self,
        train_split_path: str,
        validation_split_path: str,
        test_split_path: str,
        articles_db_path: str,
        evidence_count: int = 10,
        model_name: str = "bert-base-cased",
        max_length: int = 512,
        batch_size: int = 2
    ):
        super().__init__()

        self.train_split_path = train_split_path
        self.validation_split_path = validation_split_path
        self.test_split_path = test_split_path
        self.articles_db_path = articles_db_path
        self.evidence_count = evidence_count
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.train_class_weights = None

    @staticmethod
    def _preprocess_example(example: dict, tokenizer: PreTrainedTokenizer, max_length: int):
        preprocessed_example = { "label": torch.tensor([example["label"]]) }

        evidence_sequences = example["evidence"]

        preprocessed_example = preprocessed_example | tokenizer(
            text=[example["claim"]] * len(evidence_sequences),
            text_pair=evidence_sequences,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        return preprocessed_example

    def collate_fn(self, batch):
        collated_batch = {
            "label": torch.concat([ x["label"] for x in batch ]).view(-1, 1),
            "input_ids": torch.concat([ x["input_ids"] for x in batch ]),
            "attention_mask": torch.concat([ x["attention_mask"] for x in batch ]),
            "batch_idx": torch.concat(
                [
                    torch.tensor([idx] * len(batch[idx]["input_ids"]))
                    for idx in range(len(batch))
                ]
            ),
        }

        if "token_type_ids" in batch[0].keys():
            collated_batch["token_type_ids"] = torch.concat(
                [ x["token_type_ids"] for x in batch ]
            )

        return collated_batch

    def prepare_data(self):
        # Load the dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.tokenized_dataset = load_ctknews({
            "train": self.train_split_path,
            "validation": self.validation_split_path,
            "test": self.test_split_path
        }, k=self.evidence_count)

        # Preprocess the individual splits and assign custom fingerprints
        for split_name, split_data in self.tokenized_dataset.items():

            self.tokenized_dataset[split_name] = self.tokenized_dataset[split_name].map(
                lambda x: self._preprocess_example(x, self.tokenizer, self.max_length),
            )

        # Compute the class weights for usage in training loss
        values, counts = torch.unique(
            torch.tensor(self.tokenized_dataset["train"]["label"]), return_counts=True
        )
        self.train_class_weights = (1 / len(values)) / counts

        tensor_columns = ["input_ids", "attention_mask", "token_type_ids", "label"]
        tensor_columns = list(
            filter(
                lambda x: x in self.tokenized_dataset["train"].column_names,
                tensor_columns,
            ),
        )

        # Set the format of the dataset to PyTorch
        self.tokenized_dataset.set_format(type="torch", columns=tensor_columns)

    def get_train_class_weights(self):
        return self.train_class_weights

    def train_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["train"],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["validation"],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["test"],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )



def experiment_main(args):
    pl.seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup logging
    if not args.no_log:
        wandb_logger = WandbLogger(
            project=EXPERIMENT_NAME,
            name=f"hyperaparams_{args.n_layers}-layers_{args.n_heads}-heads_{args.connectivity}-topo_{args.hidden_dim}-hidden-{args.model_name}_{args.evidence_count}-evidence",
            log_model=True,
        )
        wandb_logger.experiment.config["batch_size"] = args.batch_size

    callbacks = [
        ModelCheckpoint(save_top_k=1, monitor="validation/f1_score", mode="max"),
    ]

    if not args.quiet:
        callbacks.append(RichProgressBar())

    ctknews = CTKNewsDataModule(
        train_split_path=RETRIEVAL_RESULTS_PATHS["train"],
        validation_split_path=RETRIEVAL_RESULTS_PATHS["validation"],
        test_split_path=RETRIEVAL_RESULTS_PATHS["test"],
        model_name=args.model_name,
        articles_db_path=ARTICLES_DB_PATH,
        evidence_count=args.evidence_count,
        batch_size=args.batch_size
    )
    ctknews.prepare_data()

    if "large" in args.model_name:
        in_features = 1024
    else:
        in_features = 768

    model = BERTPlusGNNClassifier(
        model_name=args.model_name,
        num_labels=3,
        out_features=3,
        hidden_features=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads_gnn=args.n_heads,
        connectivity=args.connectivity,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        dropout=args.dropout,
        in_features=in_features,
        log_f1=True,
        f1_aggr="micro",
        add_positional_encoding=args.evidence_importance
    )

    model.set_train_class_weights(ctknews.get_train_class_weights().to(device))

    if not args.no_log:
        if not args.skip_init_eval:
            validation_metrics = evaluate_model(model, ctknews.val_dataloader())

            wandb_logger.experiment.log(
                {
                    "validation/loss": validation_metrics[0],
                    "validation/accuracy": validation_metrics[1],
                    "validation/f1_score": -1,
                    "trainer/global_step": 0,
                }
            )

    # TODO: Make sure that the model saves the best checkpoint
    trainer = Trainer(
        default_root_dir=f"./experiments/checkpoints/{EXPERIMENT_NAME}",
        logger=False if args.no_log else wandb_logger,
        callbacks=callbacks,
        accelerator="auto",
        gpus=1 if torch.cuda.is_available() else 0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        min_epochs=min(args.min_epochs, args.max_epochs),
        max_epochs=max(args.min_epochs, args.max_epochs),
        val_check_interval=0.25,
        log_every_n_steps=10,
        enable_progress_bar=not args.quiet,
    )

    trainer.fit(model, ctknews)

    if not args.no_log:
        wandb_logger.experiment.config["epochs"] = trainer.current_epoch

    trainer.test(ckpt_path="best", dataloaders=ctknews)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", default="deepset/xlm-roberta-base-squad2", type=str)
    parser.add_argument("--evidence-count", default=10, type=int)
    parser.add_argument("--evidence-importance", action="store_true")
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--accumulate-grad-batches", default=16, type=int)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--warmup-epochs", default=0, type=int)
    parser.add_argument("--min-epochs", default=10, type=int)
    parser.add_argument("--max-epochs", default=10, type=int)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--skip-init-eval", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", default=42, type=int)

    # MODEL PARAMETERS
    parser.add_argument("--n-layers", default=1, type=int)  # 1, 2, 3, 4
    parser.add_argument("--n-heads", default=1, type=int)  # 1, 2, 4
    parser.add_argument("--hidden-dim", default=256, type=int)  # 128, 256
    parser.add_argument(
        "--connectivity", default="full", type=str
    )  # full, local_plus_global
    parser.add_argument("--dropout", default=0.5, type=float)

    args = parser.parse_args()

    experiment_main(args)
