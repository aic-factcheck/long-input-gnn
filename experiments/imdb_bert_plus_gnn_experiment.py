import torch
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
from datasets.fingerprint import Hasher
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from rich.progress import track, Progress
from typing import Tuple

import afc
from afc.models import BERTPlusGNNClassifier, GNNClassifier

EXPERIMENT_NAME = "imdb-bert-plus-gnn"

# TODO: Save the predictions for inspecting the confusion matrix of different types of examples


class IMDbDataModule(LightningDataModule):
    def __init__(
        self,
        model_name: str = "bert-base-cased",
        max_length: int = 128,
        batch_size: int = 32,
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

    @staticmethod
    def _preprocess_example(example, tokenizer, max_length):
        preprocessed_example = {"label": torch.tensor([int(example["label"])])}

        # sentences = example["sentences"]
        # sentences = afc.utils.dataset.sliding_window_sentences(
        #     sentences=example["sentences"],
        #     window_size=1
        # )

        preprocessed_example = preprocessed_example | tokenizer(
            example["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
        )

        # print(preprocessed_example)

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
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenized_dataset = load_dataset("imdb", split=["train", "test"])

        train_set, validation_set = train_test_split(
            self.tokenized_dataset[0],
            stratify=self.tokenized_dataset[0]["label"],
            test_size=2500,
        )

        train_set = Dataset.from_dict(train_set)
        validation_set = Dataset.from_dict(validation_set)

        self.tokenized_dataset = DatasetDict(
            {
                "train": train_set,
                "validation": validation_set,
                "test": self.tokenized_dataset[1],
            }
        )

        # Preprocess the individual splits and assign custom fingerprints
        for split_name, split_data in self.tokenized_dataset.items():

            new_fingerprint = Hasher.hash(
                split_data._fingerprint
                + self.model_name
                + str(self.max_length)
                + Hasher.hash(self._preprocess_example)
            )

            columns_to_remove = []
            columns_to_remove = list(
                filter(
                    lambda x: x in self.tokenized_dataset[split_name].column_names,
                    columns_to_remove,
                )
            )

            self.tokenized_dataset[split_name] = self.tokenized_dataset[split_name].map(
                lambda x: self._preprocess_example(x, tokenizer, self.max_length),
                new_fingerprint=new_fingerprint,
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
            shuffle=True,
            collate_fn=self.collate_fn,
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


def evaluate_model(
    model: torch.nn.Module, dataloader: DataLoader, display_progress=True
) -> Tuple[float, float]:
    model.eval()
    total_loss, total_accuracy = 0, 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)

    with torch.no_grad(), Progress() as progress:
        if display_progress:
            task = progress.add_task("Evaluating the model...", total=len(dataloader))

        for batch_idx, batch in enumerate(dataloader):
            batch = {
                k: v.to(dev) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_loss, batch_accuracy, _ = model._common_step(
                batch, batch_idx, stage="train"
            )
            total_loss += batch_loss.item()
            total_accuracy += batch_accuracy

            if display_progress:
                progress.update(task, advance=1)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    return avg_loss, avg_accuracy


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

    imdb = IMDbDataModule(
        max_length=128,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    imdb.prepare_data()

    model = BERTPlusGNNClassifier(
        model_name=args.model_name,
        num_labels=2,
        hidden_features=256,
        n_layers=3,
        n_heads_gnn=2,
        connectivity=args.connectivity,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
    )

    if not args.no_log:
        if not args.skip_init_eval:
            # train_metrics = evaluate_model(model, imdb.train_dataloader())
            validation_metrics = evaluate_model(model, imdb.val_dataloader())
            wandb_logger.experiment.log(
                {
                    # "train/loss_epoch": train_metrics[0],
                    # "train/accuracy_epoch": train_metrics[1],
                    "validation/loss": validation_metrics[0],
                    "validation/accuracy": validation_metrics[1],
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

    trainer.fit(model, imdb)

    if not args.no_log:
        wandb_logger.experiment.config["epochs"] = trainer.current_epoch

    trainer.test(ckpt_path="best", dataloaders=imdb)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", default="bert-base-cased", type=str)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--connectivity", default="full", type=str)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--accumulate-grad-batches", default=16, type=int)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--warmup-epochs", default=0, type=int)
    parser.add_argument("--min-epochs", default=30, type=int)
    parser.add_argument("--max-epochs", default=30, type=int)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--skip-init-eval", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", default=42, type=int)

    parser = GNNClassifier.add_model_args(parser, prefix="gnn.")

    args = parser.parse_args()

    experiment_main(args)
