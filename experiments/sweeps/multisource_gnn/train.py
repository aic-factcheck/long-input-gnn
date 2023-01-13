import os
import argparse
import torch
import wandb
import afc
import pytorch_lightning as pl
import torch_geometric as geom
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datasets.fingerprint import Hasher
from afc.models import GNNClassifier
from afc import utils

DEFAULT_CONFIG = {
    "dataset_path": "/home/hlavava1/afc/datasets/multisource/embeddings/multisource-20k-similar-medium-embedded",
    "batch_size": 32,
    "dropout": 0.5,
    "in_features": 768,
    "hidden_features": 64,
    "out_features": 2,
    "n_layers": 4,
    "n_heads": 4,
    "pooling": "max",
    "use_sag_pooling": False,
    "share_weights:": False,
    "learning_rate": 0.00002,
    "epochs": 50,
}


class MultiSourceDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        connectivity: str = "full",
        batch_size: int = 32,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.connectivity = connectivity
        self.batch_size = batch_size

    @staticmethod
    def _preprocess_example(example, connectivity):
        preprocessed_example = utils.convert_to_graph(
            nodes=[embedding[0] for embedding in example["sentences_embedding"]],
            label=example["label"],
            connectivity=connectivity,
            return_dict=True,
        )

        preprocessed_example["num_nodes"] = len(example["sentences_embedding"])

        return preprocessed_example

    def prepare_data(self, keep_progress_bar=True):
        assert os.path.exists(
            self.dataset_path
        ), "The requested dataset configuration was not found"

        self.dataset = dataset = afc.data.multisource.load_multisource(
            self.dataset_path, with_embedding=True
        )

        for split_name, split_data in self.dataset.items():
            new_fingerprint = Hasher.hash(
                split_data._fingerprint
                + self.connectivity
                + Hasher.hash(self._preprocess_example)
            )

            self.dataset[split_name] = self.dataset[split_name].map(
                lambda x: self._preprocess_example(x, self.connectivity),
                new_fingerprint=new_fingerprint,
            )

        self.dataset.set_format(
            type="torch", columns=["x", "edge_index", "y", "num_nodes"]
        )

        self.dataset.set_transform(
            lambda example: {
                "x": torch.tensor(example["x"]),
                "edge_index": torch.tensor(example["edge_index"]),
                "y": torch.tensor(example["y"]),
            },
            columns=["x", "edge_index", "y"],
        )

    def train_dataloader(self):
        return geom.loader.DataLoader(
            [geom.data.Data.from_dict(x) for x in self.dataset["train"]],
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return geom.loader.DataLoader(
            [geom.data.Data.from_dict(x) for x in self.dataset["validation"]],
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return geom.loader.DataLoader(
            [geom.data.Data.from_dict(x) for x in self.dataset["test"]],
            batch_size=self.batch_size,
        )


def train():
    wandb.init(config=DEFAULT_CONFIG)
    config = wandb.config

    multisource = MultiSourceDataModule(config.dataset_path)

    model = GNNClassifier(
        in_features=config.in_features,
        hidden_features=config.hidden_features,
        out_features=config.out_features,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        use_sag_pooling=config.use_sag_pooling,
        pooling=config.pooling,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
    )

    early_stop_callback = EarlyStopping(
        monitor="validation/loss", min_delta=0.01, patience=4
    )

    wandb_logger = WandbLogger(project="multisource-gnn")
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
