import os
import torch
import pytorch_lightning as pl
import torch_geometric as geom
from argparse import ArgumentParser
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import random_split
from rich.progress import track

from afc.models import GNNClassifier
from afc import utils as afc_utils

EXPERIMENT_NAME = "hotpotqa-transformed-gnn"


class HotpotDataModule(LightningDataModule):
    def __init__(
        self, connectivity: str = "full", batch_size: int = 32, encoding_layers: int = 4
    ):
        super().__init__()
        self.encoding_layers = encoding_layers
        self.connectivity = connectivity
        self.batch_size = batch_size

    def _get_nodes(self, example):
        nodes = [example["question_encoded"][0]]
        for document in example["context"]["sentences_encoded"]:
            nodes += document

        return nodes

    def prepare_data(self, keep_progress_bar=True):
        assert os.path.exists(
            f"./datasets/hotpotqa_by_doc_512_hidden_{self.encoding_layers}_mean"
        ), "The requested dataset configuration was not found"

        self.data = DatasetDict.load_from_disk(
            f"./datasets/hotpotqa_by_doc_512_hidden_{self.encoding_layers}_mean"
        )

        self.train_set = [
            afc_utils.convert_to_graph(
                nodes=self._get_nodes(example),
                label=example["answer"],
                connectivity=self.connectivity,
            )
            for example in track(
                self.data["train"],
                transient=not keep_progress_bar,
                description="Generating training graphs...",
            )
        ]

        self.val_set = [
            afc_utils.convert_to_graph(
                nodes=self._get_nodes(example),
                label=example["answer"],
                connectivity=self.connectivity,
            )
            for example in track(
                self.data["validation"],
                transient=not keep_progress_bar,
                description="Generating testing graphs...",
            )
        ]

        self.test_set = [
            afc_utils.convert_to_graph(
                nodes=self._get_nodes(example),
                label=example["answer"],
                connectivity=self.connectivity,
            )
            for example in track(
                self.data["test"],
                transient=not keep_progress_bar,
                description="Generating testing graphs...",
            )
        ]

    def train_dataloader(self):
        return geom.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return geom.data.DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return geom.data.DataLoader(self.test_set, batch_size=self.batch_size)


def experiment_main(args):

    pl.seed_everything(42)

    # Callbacks
    callbacks = [
        ModelCheckpoint(save_top_k=3, monitor="validation/accuracy", mode="max"),
        EarlyStopping(
            monitor="train/accuracy",
            mode="max",
            patience=10,
        ),
        RichProgressBar(),
    ]

    # Define the model
    model = GNNClassifier(
        args.in_features,
        args.hidden_features,
        args.out_features,
        args.n_layers,
        args.dropout,
        args.lr,
    )

    # Logging
    if not args.no_log:
        wandb_logger = WandbLogger(project=EXPERIMENT_NAME, log_model="all")
        # log the gradients and model topology
        wandb_logger.watch(model)

        wandb_logger.experiment.config["graph_connectivity"] = args.connectivity
        wandb_logger.experiment.config["batch_size"] = args.batch_size
        wandb_logger.experiment.config["dropout"] = args.dropout
        wandb_logger.experiment.config["encoding_layers"] = args.encoding_layers

    hotpot = HotpotDataModule(
        connectivity=args.connectivity,
        batch_size=args.batch_size,
        encoding_layers=args.encoding_layers,
    )

    trainer = Trainer(
        default_root_dir=f"./experiments/checkpoints/{EXPERIMENT_NAME}",
        logger=False if args.no_log else wandb_logger,
        callbacks=callbacks,
        accelerator="auto",
        gpus=1 if torch.cuda.is_available() else 0,
        min_epochs=20,
    )

    trainer.fit(model, hotpot)

    # Log the number of epochs for which the model has been trained
    if not args.no_log:
        wandb_logger.experiment.config["epochs"] = trainer.current_epoch

    # Calling trainer.test after trainer.fit defaults to
    # using the best attained checkpoint
    # trainer.test(test_loader) # did not work properly, the line below fixed it
    trainer.test(ckpt_path="best", dataloaders=hotpot)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-features", default=768, type=int)
    parser.add_argument("--hidden-features", default=512, type=int)
    parser.add_argument("--out-features", default=2, type=int)
    parser.add_argument("--n-layers", default=3, type=int)
    parser.add_argument("--lr", default=0.00002, type=float)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--connectivity", default="full", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument(
        "--encoding-layers",
        default=4,
        type=int,
        help="number of the BERT's last hidden layers used to generate the encoding",
    )
    parser.add_argument("--no-log", action="store_true")

    args = parser.parse_args()

    experiment_main(args)
