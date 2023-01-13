import torch
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset, disable_caching
from datasets.fingerprint import Hasher
from torch.utils.data import DataLoader
from rich.progress import track, Progress
from typing import Tuple

import afc
from afc.models import BERTPlusGNNClassifier, GNNClassifier
from afc.models.utils import evaluate_model

EXPERIMENT_NAME = "multisource-ratio-influence"

disable_caching()


class MultiSourceDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str = None,
        model_name: str = "bert-base-cased",
        max_length: int = 512,
        window_size: int = 1,
        batch_size: int = 2,
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.model_name = model_name
        self.max_length = max_length
        self.window_size = window_size
        self.batch_size = batch_size

    @staticmethod
    def _preprocess_example(example, tokenizer, max_length, window_size):
        preprocessed_example = {"label": torch.tensor([int(example["label"])])}

        sequences = afc.utils.dataset.sliding_window_sentences(
            example["sentences"], window_size=window_size
        )

        preprocessed_example = preprocessed_example | tokenizer(
            sequences,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        return preprocessed_example

    def collate_fn(self, batch):
        collated_batch = {
            "label": torch.concat([x["label"] for x in batch]).view(-1, 1),
            "input_ids": torch.concat([x["input_ids"] for x in batch]),
            "attention_mask": torch.concat([x["attention_mask"] for x in batch]),
            "batch_idx": torch.concat(
                [
                    torch.tensor([idx] * len(batch[idx]["input_ids"]))
                    for idx in range(len(batch))
                ]
            ),
        }

        if "token_type_ids" in batch[0].keys():
            collated_batch["token_type_ids"] = torch.concat([x["token_type_ids"] for x in batch])

        return collated_batch

    def prepare_data(self):
        # Load the dataset
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenized_dataset = afc.data.multisource.load_multisource(
            self.dataset_path
        )

        # Preprocess the individual splits and assign custom fingerprints
        for split_name, split_data in self.tokenized_dataset.items():

            new_fingerprint = Hasher.hash(
                split_data._fingerprint
                + self.model_name
                + str(self.max_length)
                + str(self.window_size)
                + Hasher.hash(self._preprocess_example)
            )

            columns_to_remove = ["ids", "sentences"]
            columns_to_remove = list(
                filter(
                    lambda x: x in self.tokenized_dataset[split_name].column_names,
                    columns_to_remove,
                )
            )

            self.tokenized_dataset[split_name] = self.tokenized_dataset[split_name].map(
                lambda x: self._preprocess_example(x, tokenizer, self.max_length, self.window_size),
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


def experiment_main(args):
    pl.seed_everything(args.seed)

    if args.dataset_path[-1] == "/":
        args.dataset_path = args.dataset_path[:-1]
    dataset_name = args.dataset_path.split("/")[-1]

    # Setup logging
    if not args.no_log:
        wandb_logger = WandbLogger(
            project=EXPERIMENT_NAME,
            name=f"{dataset_name}_{args.window_size}-window_{args.connectivity}-topo",
            log_model=True
        )
        wandb_logger.experiment.config["batch_size"] = args.batch_size

    callbacks = [
        ModelCheckpoint(save_top_k=1, monitor="validation/accuracy", mode="max"),
    ]

    if not args.quiet:
        callbacks.append(RichProgressBar())

    multisource = MultiSourceDataModule(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        window_size=args.window_size,
        batch_size=args.batch_size,
    )
    multisource.prepare_data()

    model = BERTPlusGNNClassifier(
        model_name=args.model_name,
        num_labels=2,
        hidden_features=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads_gnn=args.n_heads,
        connectivity=args.connectivity,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
    )

    if not args.no_log:
        if not args.skip_init_eval:
            validation_metrics = evaluate_model(model, multisource.val_dataloader())

            wandb_logger.experiment.log(
                {
                    "validation/loss": validation_metrics[0],
                    "validation/accuracy": validation_metrics[1],
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

    trainer.fit(model, multisource)

    if not args.no_log:
        wandb_logger.experiment.config["epochs"] = trainer.current_epoch

    trainer.test(ckpt_path="best", dataloaders=multisource)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--model-name", default="bert-base-cased", type=str)
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
    parser.add_argument("--n-layers", default=4, type=int)              # 1, 2, 3, 4
    parser.add_argument("--n-heads", default=2, type=int)               # 1, 2, 4
    parser.add_argument("--hidden-dim", default=256, type=int)          # 128, 256
    parser.add_argument("--connectivity", default="full", type=str)     # full, local+global
    parser.add_argument("--window-size", default=1, type=int)           # 1, 2, 3, 4

    args = parser.parse_args()

    experiment_main(args)
