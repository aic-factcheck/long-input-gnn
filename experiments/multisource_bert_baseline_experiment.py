import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
from datasets.fingerprint import Hasher
from torch.utils.data import DataLoader
from rich.progress import track

import afc
from afc.models import BERTClassifier
from afc.models.utils import evaluate_model

EXPERIMENT_NAME = "multisource-bert-baseline"

# TODO: Save the predictions for inspecting the confusion matrix of different types of examples


class MultiSourceDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str = None,
        model_name: str = "bert-base-cased",
        max_length: int = 512,
        include_title: bool = False,
        batch_size: int = 32,
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

    @staticmethod
    def _preprocess_example(example, tokenizer, max_length):
        preprocessed_example = {"label": torch.tensor([int(example["label"])])}

        preprocessed_example = preprocessed_example | tokenizer(
            " ".join(example["sentences"]),
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        return preprocessed_example

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
                + Hasher.hash(self._preprocess_example)
            )

            self.tokenized_dataset[split_name] = self.tokenized_dataset[split_name].map(
                lambda x: self._preprocess_example(x, tokenizer, self.max_length),
                new_fingerprint=new_fingerprint,
                remove_columns=["ids", "sentences"],
            )

        # Set the format of the dataset to PyTorch
        self.tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "token_type_ids", "label"],
        )

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
        return DataLoader(self.tokenized_dataset["test"], batch_size=self.batch_size)


def experiment_main(args):
    pl.seed_everything(args.seed)

    # Setup logging
    if not args.no_log:
        wandb_logger = WandbLogger(project=EXPERIMENT_NAME, log_model=True)
        wandb_logger.experiment.config["batch_size"] = args.batch_size

    callbacks = [
        ModelCheckpoint(save_top_k=1, monitor="validation/accuracy", mode="max"),
        # EarlyStopping(
        #     monitor="validation/accuracy",
        #     mode="max",
        #     min_delta=0.01,
        #     patience=3,
        # ),
    ]

    if not args.quiet:
        callbacks.append(RichProgressBar())

    multisource = MultiSourceDataModule(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
    )

    model = BERTClassifier(num_labels=2, learning_rate=args.lr)

    if not args.no_log:
        wandb_logger.watch(model)

        if not args.skip_init_eval:
            validation_metrics = evaluate_model(model, multisource.val_dataloader())

            wandb_logger.experiment.log(
                {
                    "validation/loss": validation_metrics[0],
                    "validation/accuracy": validation_metrics[1],
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

    trainer.fit(model, multisource)

    if not args.no_log:
        wandb_logger.experiment.config["epochs"] = trainer.current_epoch

    trainer.test(ckpt_path="best", dataloaders=multisource)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-path", default="./datasets/multisource-50k", type=str
    )
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--min-epochs", default=5, type=int)
    parser.add_argument("--max-epochs", default=10, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-init-eval", action="store_true")
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    experiment_main(args)
