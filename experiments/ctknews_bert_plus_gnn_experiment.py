import torch
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from sklearn import metrics
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
from datasets.fingerprint import Hasher
from torch.utils.data import DataLoader
from rich.progress import track, Progress
from typing import Tuple

import afc
from afc.data.ctknews import load_ctknews
from afc.models import BERTPlusGNNClassifier, GNNClassifier
from afc.models.utils import evaluate_model, get_model_predictions

EXPERIMENT_NAME = "ctknews-bert-plus-gnn"

# TODO: Save the predictions for inspecting the confusion matrix of different types of examples
# NOTE: One way to work around the end-to-end training memory requirements is to fix certain
#       encodings and backpropagate only through a random subset of the nodes

class CTKNewsDataModule(LightningDataModule):
    def __init__(
        self,
        train_split_path: str,
        validation_split_path: str,
        test_split_path: str,
        articles_db_path: str,
        evidence_count: int = 20,
        model_name: str = "bert-base-cased",
        max_length: int = 512,
        batch_size: int = 1
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

    @staticmethod
    def _preprocess_example(example: dict, tokenizer, max_length: int):
        preprocessed_example = { "label": torch.tensor([example["label"]]) }

        # evidence_sequences = afc.utils.dataset.sliding_window_sentences(
        #     sentences=example["evidence"], window_size=2
        # )

        evidence_sequences = example["evidence"]

        # Encoding each evidence part as "claim"[SEP]"evidence"
        preprocessed_example = preprocessed_example | tokenizer(
            text=[example["claim"]] * len(evidence_sequences),
            text_pair=evidence_sequences,
            max_length=max_length,
            truncation=True,
            padding="longest",
        )

        return preprocessed_example

    def collate_fn(self, batch):
        collated_batch = {
            "label": torch.concat([x["label"] for x in batch]).view(-1, 1),
            "input_ids": torch.concat([torch.vstack(x["input_ids"]) for x in batch]),
            "attention_mask": torch.concat(
                [torch.vstack(x["attention_mask"]) for x in batch]
            ),
            "batch_idx": torch.concat(
                [
                    torch.tensor([idx] * len(batch[idx]["input_ids"]))
                    for idx in range(len(batch))
                ]
            ),
        }

        if "token_type_ids" in batch[0].keys():
            collated_batch["token_type_ids"] = torch.concat(
                [torch.vstack(x["token_type_ids"]) for x in batch]
            )

        return collated_batch

    def prepare_data(self):
        # Load the dataset
        print("Getting the tokenizer...", end=" ")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("done!")

        self.tokenized_dataset = load_ctknews({
            "train": self.train_split_path,
            "validation": self.validation_split_path,
            "test": self.test_split_path
        }, k=self.evidence_count, articles_db_path=self.articles_db_path)

        # Preprocess the individual splits and assign custom fingerprints
        for split_name, split_data in self.tokenized_dataset.items():
            columns_to_remove = ["ids", "sentences"]
            columns_to_remove = list(
                filter(
                    lambda x: x in self.tokenized_dataset[split_name].column_names,
                    columns_to_remove,
                )
            )

            self.tokenized_dataset[split_name] = self.tokenized_dataset[split_name].map(
                lambda x: self._preprocess_example(x, tokenizer, self.max_length),
                remove_columns=columns_to_remove,
            )

        tensor_columns = ["input_ids", "attention_mask", "token_type_ids", "label"]
        tensor_columns = list(
            filter(
                lambda x: x in self.tokenized_dataset["train"].column_names,
                tensor_columns,
            ),
        )

        # Set the format of the dataset to PyTorch
        self.tokenized_dataset.set_format(type="torch", columns=tensor_columns)

    def train_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["train"],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True
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
            collate_fn=self.collate_fn,
        )


def experiment_main(args):
    pl.seed_everything(args.seed)

    # Setup logging
    if not args.no_log:
        wandb_logger = WandbLogger(project=EXPERIMENT_NAME, log_model=True)
        wandb_logger.experiment.config["batch_size"] = args.batch_size

    callbacks = [
        ModelCheckpoint(save_top_k=2, monitor="validation/accuracy", mode="max"),
    ]

    if not args.quiet:
        callbacks.append(RichProgressBar())

    ctknews = CTKNewsDataModule(
        train_split_path=args.train_split_path,
        validation_split_path=args.validation_split_path,
        test_split_path=args.test_split_path,
        articles_db_path=args.articles_db_path,
        evidence_count=args.evidence_count,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    ctknews.prepare_data()

    # Check the dimensionality of the input features at runtime to support
    # different models, e.g. bert-base-cased (768) vs. bert-large-cased (1024)
    # currently does not work ("max_length" vs "longest")
    # in_features = ctknews.tokenized_dataset["train"][0]["input_ids"].shape[1]

    model = BERTPlusGNNClassifier(
        model_name=args.model_name,
        num_labels=3,
        in_features=768,
        out_features=3,
        hidden_features=256,
        n_layers=2,
        n_heads_gnn=4,
        connectivity=args.connectivity,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
    )

    if not args.no_log:
        if not args.skip_init_eval:
            validation_metrics = evaluate_model(model, ctknews.val_dataloader())
            predictions = get_model_predictions(model, ctknews.val_dataloader())
            f1 = metrics.f1_score(ctknews.tokenized_dataset["validation"]["label"], predictions.cpu())

            wandb_logger.experiment.log(
                {
                    "validation/loss": validation_metrics[0],
                    "validation/accuracy": validation_metrics[1],
                    "validation/f1": f1,
                    "trainer/global_step": 0,
                }
            )

        wandb_logger.watch(model)

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
    parser.add_argument("--train-split-path", type=str)
    parser.add_argument("--validation-split-path", type=str)
    parser.add_argument("--test-split-path", type=str)
    parser.add_argument("--articles-db-path", type=str)
    parser.add_argument("--evidence-count", default=10, type=int)
    parser.add_argument("--model-name", default="bert-base-cased", type=str)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--accumulate-grad-batches", default=16, type=int)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--warmup-epochs", default=0, type=int)
    parser.add_argument("--connectivity", default="full", type=str)
    parser.add_argument("--min-epochs", default=30, type=int)
    parser.add_argument("--max-epochs", default=30, type=int)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--skip-init-eval", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", default=42, type=int)

    parser = GNNClassifier.add_model_args(parser, prefix="gnn.")

    args = parser.parse_args()

    experiment_main(args)
