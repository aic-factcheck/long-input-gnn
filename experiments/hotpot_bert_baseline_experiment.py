import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from rich.progress import track

import afc
from afc.models import BERTClassifier
from afc import utils as afc_utils

EXPERIMENT_NAME = "hotpotqa-transformed-bert-baseline"


class HotpotDataModule(LightningDataModule):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        use_sep: bool = False,
        supporting_first: bool = False,
        oracle: bool = False,
        include_title: bool = False,
        batch_size: int = 32,
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.use_sep = use_sep
        self.supporting_first = supporting_first
        self.oracle = oracle
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.oracle:
            self.supporting_first = False
            print(
                "[Dataset] Running in oracle mode - providing only sentences that contain the supporting facts"
            )

        if self.supporting_first:
            print("[Dataset] Supporting documents are prioritized")

        if self.use_sep:
            print("[Dataset] Encoding uses the [SEP] token")

    def _preprocess_example(self, example):
        preprocessed_example = {
            "label": torch.tensor([example["answer"]]),
        }

        if self.oracle:
            documents_to_use = [
                example["context"]["sentences"][
                    example["context"]["title"].index(title)
                ][sentence_idx]
                for title, sentence_idx in zip(*example["supporting_facts"].values())
            ]
        elif self.supporting_first:
            # Provide the articles known to support the statement as first
            supporting_idxs = [
                example["context"]["title"].index(title)
                for title in set(example["supporting_facts"]["title"])
            ]
            other_idxs = filter(
                lambda x: x not in supporting_idxs,
                range(len(example["context"]["title"])),
            )

            documents_to_use = [
                example["context"]["sentences"][idx] for idx in supporting_idxs
            ]
            documents_to_use += [
                example["context"]["sentences"][idx] for idx in other_idxs
            ]
        else:
            # Otherwise just use the order in which they were provided
            documents_to_use = example["context"]["sentences"]

        documents = " ".join(["".join(document) for document in documents_to_use])
        # Compute the average length of the documents over all examples to see whether
        # the oracle setting works correctly

        if self.use_sep:
            preprocessed_example = preprocessed_example | self.tokenizer(
                example["question"],
                documents,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )
        else:
            preprocessed_example = preprocessed_example | self.tokenizer(
                example["question"] + " " + documents,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )

        return preprocessed_example

    def prepare_data(self):
        # Load the dataset
        dataset = afc.data.hotpotqa.load_hotpotqa()

        # Try overfitting the training dataset
        # dataset["train"] = dataset["train"].select(range(100))
        # dataset["validation"] = dataset["validation"].select(range(100))

        # Transform the dataset (construct input sequences and pre-tokenize them)
        self.tokenized_dataset = dataset.map(self._preprocess_example)
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
    pl.seed_everything(42)

    # Setup logging
    if not args.no_log:
        # run_name = f"{EXPERIMENT_NAME}_lr={args.lr}_bs={args.batch_size}"
        # if args.use_sep:
        #     run_name += "_sep"
        # if args.oracle:
        #     run_name += "_oracle-sentences"
        # elif args.supporting_first:
        #     run_name += "_oracle-docs"

        wandb_logger = WandbLogger(project=EXPERIMENT_NAME, log_model=True)
        wandb_logger.experiment.config["batch_size"] = args.batch_size
        wandb_logger.experiment.config["use_sep"] = args.use_sep
        wandb_logger.experiment.config["supporting_first"] = args.supporting_first
        wandb_logger.experiment.config["oracle"] = args.oracle

    callbacks = [
        ModelCheckpoint(save_top_k=1, monitor="validation/accuracy", mode="max"),
        EarlyStopping(
            monitor="validation/accuracy",
            mode="max",
            min_delta=0.01,
            patience=3,
        ),
        RichProgressBar(),
    ]

    hotpot = HotpotDataModule(
        use_sep=args.use_sep,
        supporting_first=args.supporting_first,
        oracle=args.oracle,
        batch_size=args.batch_size,
    )

    model = BERTClassifier(num_labels=2, learning_rate=args.lr)

    if not args.no_log:
        wandb_logger.watch(model)

    trainer = Trainer(
        default_root_dir=f"./experiments/checkpoints/{EXPERIMENT_NAME}",
        logger=False if args.no_log else wandb_logger,
        callbacks=callbacks,
        accelerator="auto",
        gpus=1 if torch.cuda.is_available() else 0,
        min_epochs=10,
        max_epochs=20,
    )

    trainer.fit(model, hotpot)

    if not args.no_log:
        wandb_logger.experiment.config["epochs"] = trainer.current_epoch

    trainer.test(ckpt_path="best", dataloaders=hotpot)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--supporting-first",
        action="store_true",
        help="present the documents containing supporting facts as first",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="present only the sentences containing the supporting facts",
    )
    parser.add_argument(
        "--use-sep",
        action="store_true",
        help="encode the question and documents using the [SEP] token",
    )
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--no-log", action="store_true")

    args = parser.parse_args()

    args.supporting_first = False if args.oracle else args.supporting_first

    experiment_main(args)
