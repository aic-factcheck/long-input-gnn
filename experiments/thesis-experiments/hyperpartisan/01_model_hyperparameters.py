import torch
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_from_disk
from datasets.fingerprint import Hasher
from torch.utils.data import DataLoader
from rich.progress import track, Progress
from typing import Tuple

import afc
from afc.models import BERTPlusGNNClassifier, GNNClassifier
from afc.models.utils import evaluate_model

EXPERIMENT_NAME = "hyperpartisan-model-hyperparams"


class HyperpartisanDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        model_name: str = "bert-base-cased",
        max_length: int = 512,
        batch_size: int = 1,
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

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
            collated_batch["token_type_ids"] = torch.concat(
                [x["token_type_ids"] for x in batch]
            )

        return collated_batch

    def prepare_data(self):
        # Load the dataset
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.tokenized_dataset = load_from_disk(self.dataset_path)

        self.tokenized_dataset = self.tokenized_dataset.map(
            lambda example: tokenizer.encode_plus(
                example["text"],
                max_length=512,
                padding=True,
                truncation=True,
                return_overflowing_tokens=True,
            )
            | {"label": [example["label"]]}
        )

        # Compute the class weights for usage in training loss
        values, counts = torch.unique(
            torch.tensor(self.tokenized_dataset["train"]["label"]), return_counts=True
        )
        self.train_set_class_weights = (1 / len(values)) / counts

        # Set the format of the dataset to PyTorch
        tensor_columns = ["input_ids", "attention_mask", "token_type_ids", "label"]
        tensor_columns = [ x for x in tensor_columns if x in self.tokenized_dataset["train"].column_names ]
        self.tokenized_dataset.set_format(type="torch", columns=tensor_columns)

    def get_train_class_weights(self):
        return self.train_set_class_weights

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

    print(f"Using dataset: {args.dataset_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup logging
    if not args.no_log:
        wandb_logger = WandbLogger(
            project=EXPERIMENT_NAME,
            name=f"hyperaparams_{args.n_layers}-layers_{args.n_heads}-heads_{args.connectivity}-topo_{args.hidden_dim}-hidden-{args.model_name}",
            log_model=True,
        )
        wandb_logger.experiment.config["batch_size"] = args.batch_size

    callbacks = [
        ModelCheckpoint(save_top_k=1, monitor="validation/f1_score", mode="max"),
    ]

    if not args.quiet:
        callbacks.append(RichProgressBar())

    hyperpartisan = HyperpartisanDataModule(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    hyperpartisan.prepare_data()

    if "large" in args.model_name:
        in_features = 1024
    else:
        in_features = 768

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
        dropout=args.dropout,
        in_features=in_features,
        log_f1=True
    )

    model.set_train_class_weights(hyperpartisan.get_train_class_weights().to(device))

    wandb_logger.experiment.define_metric("validation/accuracy", summary="max")
    wandb_logger.experiment.define_metric("validation/f1_score", summary="max")

    if not args.no_log:
        if not args.skip_init_eval:
            validation_metrics = evaluate_model(model, hyperpartisan.val_dataloader())

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

    trainer.fit(model, hyperpartisan)

    if not args.no_log:
        wandb_logger.experiment.config["epochs"] = trainer.current_epoch

    trainer.test(ckpt_path="best", dataloaders=hyperpartisan)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-path", default="/home/hlavava1/afc/datasets/thesis-datasets/hyperpartisan", type=str
    )
    parser.add_argument("--model-name", default="bert-base-cased", type=str)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--accumulate-grad-batches", default=32, type=int)
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
