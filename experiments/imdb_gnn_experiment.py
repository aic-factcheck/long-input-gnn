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


class IMDBDataModule(LightningDataModule):
    def __init__(self, connectivity: str = "full", batch_size: int = 32):
        super().__init__()
        self.connectivity = connectivity
        self.batch_size = batch_size

    def prepare_data(self):
        self.data = DatasetDict.load_from_disk(
            "./datasets/imdb_split_256_hidden_4_mean"
        )

        full_train_set = [
            afc_utils.convert_to_graph(doc, self.connectivity)
            for doc in track(
                self.data["train"],
                transient=True,
                description="Generating training graphs...",
            )
        ]

        train_size = int(len(self.data["train"]) * 0.8)
        val_size = len(self.data["train"]) - train_size
        self.train_set, self.val_set = random_split(
            full_train_set, [train_size, val_size]
        )

        self.test_set = [
            afc_utils.convert_to_graph(doc, self.connectivity)
            for doc in track(
                self.data["test"],
                transient=True,
                description="Generating testing graphs...",
            )
        ]

    def train_dataloader(self):
        return geom.data.DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return geom.data.DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return geom.data.DataLoader(self.test_set, batch_size=self.batch_size)


def load_datasets(connectivity: str, batch_size: int):
    # Loads an already split, tokenized and encoded dataset
    # The splits have maximum length of 256 tokens
    # The encoding of a split is a mean of the values of the last 4 hidden layers of BERT
    dataset = DatasetDict.load_from_disk("./datasets/imdb_split_256_hidden_4_mean")

    # Convert the chunks into a graph
    train_set = [
        afc_utils.convert_to_graph(doc, connectivity)
        for doc in track(
            dataset["train"],
            transient=True,
            description="Generating training graphs...",
        )
    ]
    test_set = [
        afc_utils.convert_to_graph(doc, connectivity)
        for doc in track(
            dataset["test"], transient=True, description="Generating testing graphs..."
        )
    ]

    # Sizes for the training and validation sets
    train_size = int(len(dataset["train"]) * 0.8)
    val_size = len(dataset["train"]) - train_size

    train_set, val_set = random_split(train_set, [train_size, val_size])

    train_loader = geom.loader.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = geom.loader.DataLoader(val_set, batch_size=batch_size)
    test_loader = geom.loader.DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def experiment_main(args):

    pl.seed_everything(42)

    # Callbacks
    callbacks = [
        ModelCheckpoint(save_top_k=3, monitor="validation/accuracy", mode="max"),
        EarlyStopping(monitor="train/accuracy", mode="max", min_delta=0.0, patience=10),
        RichProgressBar(),
    ]

    # Define the model
    model = GNNClassifier(
        args.in_features,
        args.hidden_features,
        args.out_features,
        args.n_layers,
        args.lr,
    )

    # Logging
    if not args.no_log:
        wandb_logger = WandbLogger(project="imdb-gnn-classification", log_model="all")
        # log the gradients and model topology
        wandb_logger.watch(model)

        wandb_logger.experiment.config["graph_connectivity"] = args.connectivity
        wandb_logger.experiment.config["batch_size"] = args.batch_size

    # Get the data loaders
    # train_loader, val_loader, test_loader = load_datasets(
    #     args.connectivity, args.batch_size
    # )

    imdb = IMDBDataModule(connectivity=args.connectivity, batch_size=args.batch_size)

    trainer = Trainer(
        default_root_dir="./experiments/checkpoints",
        logger=False if args.no_log else wandb_logger,
        callbacks=callbacks,
        accelerator="auto",
        gpus=1,
    )

    # trainer.fit(model, train_loader, val_loader)
    trainer.fit(model, imdb)

    # Log the number of epochs for which the model has been trained
    if not args.no_log:
        wandb_logger.experiment.config["epochs"] = trainer.current_epoch

    # Calling trainer.test after trainer.fit defaults to
    # using the best attained checkpoint
    # trainer.test(test_loader)
    trainer.test(ckpt_path="best", dataloaders=imdb)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-features", default=768, type=int)
    parser.add_argument("--hidden-features", default=512, type=int)
    parser.add_argument("--out-features", default=2, type=int)
    parser.add_argument("--n-layers", default=3, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--connectivity", default="full", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--no-log", action="store_true")

    args = parser.parse_args()

    experiment_main(args)
