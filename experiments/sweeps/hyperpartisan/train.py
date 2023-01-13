import os
import argparse
import torch
import wandb
import afc
import pytorch_lightning as pl
import torch_geometric as geom
import datasets
from transformers import AutoTokenizer
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from afc.models import BERTPlusGNNClassifier
from afc import utils

DEFAULT_CONFIG = {
    "dataset_path": "/home/hlavava1/afc/datasets/multisource/embeddings/multisource-20k-similar-medium-embedded",
    "batch_size": 32,
    "dropout": 0.1,
    "in_features": 768,
    "hidden_features": 128,
    "out_features": 2,
    "n_layers": 4,
    "n_heads": 4,
    "pooling": "max",
    "use_sag_pooling": False,
    "share_weights:": False,
    "learning_rate": 0.00002,
    "epochs": 30,
}


class HyperpartisanDataModule(LightningDataModule):
    def __init__(
        self,
        model_name: str = "bert-base-cased",
        max_length: int = 512,
        batch_size: int = 1,
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

    def collate_fn(self, batch):
        collated_batch = {
            "label": torch.concat([x["label"] for x in batch]).view(-1, 1),
            "input_ids": torch.concat([x["input_ids"] for x in batch]),
            "attention_mask": torch.concat(
                [x["attention_mask"] for x in batch]
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
                [x["token_type_ids"] for x in batch]
            )

        return collated_batch

    def prepare_data(self):
        # Load the dataset
        tokenizer = self.tok if self.tok is not None else AutoTokenizer.from_pretrained(self.model_name)
        # self.tokenized_dataset = datasets.load_dataset("hyperpartisan_news_detection", "byarticle")["train"]
        # self.tokenized_dataset = prepare_hyperpartisan_dataset(self.tokenized_dataset)
        self.tokenized_dataset = datasets.load_from_disk("/home/hlavava1/afc/datasets/hyperpartisan")
        
        self.tokenized_dataset = self.tokenized_dataset.map(
            lambda example: tokenizer.encode_plus(
                example["text"],
                max_length=512,
                padding=True,
                truncation=True,
                return_overflowing_tokens=True
            ) | { "label": [example["label"]] }
        )

        tensor_columns = ["input_ids", "attention_mask", "token_type_ids", "label"]

        values, counts = torch.unique(torch.tensor(self.tokenized_dataset["train"]["label"]), return_counts=True)
        self.train_set_class_weights = (1 / len(values)) / counts

        # Set the format of the dataset to PyTorch
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


def train():
    wandb.init(config=DEFAULT_CONFIG)
    config = wandb.config

    multisource = HyperpartisanDataModule(config.dataset_path)

    model = BERTPlusGNNClassifier(
        in_features=config.in_features,
        hidden_features=config.hidden_features,
        out_features=config.out_features,
        n_heads_gnn=config.n_heads,
        n_layers=config.n_layers,
        pooling=config.pooling,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
    )

    early_stop_callback = EarlyStopping(
        monitor="validation/loss", min_delta=0.01, patience=4
    )

    wandb_logger = WandbLogger(project="hyperpartisan")
    trainer = Trainer(
        logger=wandb_logger,
        accelerator="auto",
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=config.epochs,
        callbacks=[early_stop_callback],
    )

    trainer.fit(model, multisource)


if __name__ == "__main__":
    train()
