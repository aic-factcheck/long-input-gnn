import os
import sqlite3
import torch
import pytorch_lightning as pl
import pandas as pd
from argparse import ArgumentParser
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from transformers import AutoTokenizer
import datasets
from datasets import Dataset, DatasetDict, load_from_disk
from datasets.fingerprint import Hasher
from torch.utils.data import DataLoader
from rich.progress import track

import afc
from afc.models import BERTClassifier
from afc.models.utils import evaluate_model

EXPERIMENT_NAME = "CTKNEWS-THESIS-BASELINES"

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
    def _preprocess_example(example: dict, tokenizer, max_length: int):
        preprocessed_example = { "label": torch.tensor([example["label"]]) }

        evidence_sequences = example["evidence"]

        preprocessed_example = preprocessed_example | tokenizer(
            text=example["claim"],
            text_pair=" ".join(evidence_sequences),
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        return preprocessed_example

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
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["validation"],
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["test"],
            batch_size=self.batch_size,
        )



def experiment_main(args):
    pl.seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup logging
    if not args.no_log:
        wandb_logger = WandbLogger(
            project=EXPERIMENT_NAME,
            name=f"ctknews_{args.model_name}_{args.evidence_count}-evidence",
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

    model = BERTClassifier(
        model_name=args.model_name,
        num_labels=3,
        learning_rate=args.lr,
        log_f1=True,
        f1_aggr="micro"
    )

    model.set_train_class_weights(
        ctknews.get_train_class_weights().to(device)
    )

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

    trainer = Trainer(
        default_root_dir=f"./experiments/checkpoints/{EXPERIMENT_NAME}",
        logger=False if args.no_log else wandb_logger,
        callbacks=callbacks,
        accelerator="auto",
        gpus=1 if torch.cuda.is_available() else 0,
        min_epochs=min(args.min_epochs, args.max_epochs),
        max_epochs=max(args.min_epochs, args.max_epochs),
        val_check_interval=0.25,
        enable_progress_bar=not args.quiet,
    )

    trainer.fit(model, ctknews)

    if not args.no_log:
        wandb_logger.experiment.config["epochs"] = trainer.current_epoch

    trainer.test(ckpt_path="best", dataloaders=ctknews)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", default="bert-base-cased", type=str)
    parser.add_argument("--evidence-count", default=10, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--min-epochs", default=10, type=int)
    parser.add_argument("--max-epochs", default=10, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-init-eval", action="store_true")
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    experiment_main(args)
