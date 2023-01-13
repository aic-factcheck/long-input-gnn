import os
import torch
import pytorch_lightning as pl
import torch_geometric as geom
from argparse import ArgumentParser
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from datasets.fingerprint import Hasher
from rich.progress import track

import afc
from afc.models import GNNClassifier
from afc import utils

EXPERIMENT_NAME = "multisource-gnn"


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


def experiment_main(args):

    pl.seed_everything(42)

    # Callbacks
    callbacks = [
        ModelCheckpoint(save_top_k=3, monitor="validation/accuracy", mode="max"),
        EarlyStopping(
            monitor="validation/accuracy",
            mode="max",
            min_delta=0.01,
            patience=10,
        ),
    ]

    if not args.quiet:
        callbacks.append(RichProgressBar())

    # Define the model
    model = GNNClassifier(
        in_features=args.in_features,
        hidden_features=args.hidden_features,
        out_features=args.out_features,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        share_weights=args.share_weights,
        pooling=args.pooling,
        dropout=args.dropout,
        learning_rate=args.lr,
    )

    # Logging
    if not args.no_log:
        wandb_logger = WandbLogger(project=EXPERIMENT_NAME, log_model="all")
        # log the gradients and model topology
        wandb_logger.watch(model)

        wandb_logger.experiment.config["dataset_path"] = args.dataset_path
        wandb_logger.experiment.config["graph_connectivity"] = args.connectivity
        wandb_logger.experiment.config["batch_size"] = args.batch_size
        wandb_logger.experiment.config["dropout"] = args.dropout

    multisource = MultiSourceDataModule(
        dataset_path=args.dataset_path,
        connectivity=args.connectivity,
        batch_size=args.batch_size,
    )

    trainer = Trainer(
        default_root_dir=f"./experiments/checkpoints/{EXPERIMENT_NAME}",
        logger=False if args.no_log else wandb_logger,
        callbacks=callbacks,
        accelerator="auto",
        gpus=1 if torch.cuda.is_available() else 0,
        enable_progress_bar=not args.quiet,
        min_epochs=min(args.min_epochs, args.max_epochs),
        max_epochs=max(args.min_epochs, args.max_epochs),
    )

    trainer.fit(model, multisource)

    # Log the number of epochs for which the model has been trained
    if not args.no_log:
        wandb_logger.experiment.config["epochs"] = trainer.current_epoch

    trainer.test(ckpt_path="best", dataloaders=multisource)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--in-features", default=768, type=int)
    parser.add_argument("--hidden-features", default=512, type=int)
    parser.add_argument("--out-features", default=2, type=int)
    parser.add_argument("--n-heads", default=1, type=int)
    parser.add_argument("--n-layers", default=3, type=int)
    parser.add_argument("--share-weights", action="store_true")
    parser.add_argument("--pooling", default="max", type=str, choices=["mean", "max"])
    parser.add_argument("--lr", default=0.00002, type=float)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--connectivity", default="full", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--min-epochs", default=20, type=int)
    parser.add_argument("--max-epochs", default=30, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-log", action="store_true")

    args = parser.parse_args()

    experiment_main(args)
