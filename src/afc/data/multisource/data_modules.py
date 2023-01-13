import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from datasets.fingerprint import Hasher
import afc


class MultiSourceDataModuleForBertPlusGnn(LightningDataModule):
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
            example["sentences"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        return preprocessed_example

    def collate_fn(self, batch):
        label = torch.concat([x["label"] for x in batch]).view(-1, 1)
        input_ids = torch.concat([torch.vstack(x["input_ids"]) for x in batch])
        token_type_ids = torch.concat(
            [torch.vstack(x["token_type_ids"]) for x in batch]
        )
        attention_mask = torch.concat(
            [torch.vstack(x["attention_mask"]) for x in batch]
        )
        batch_idx = torch.concat(
            [
                torch.tensor([idx] * len(batch[idx]["input_ids"]))
                for idx in range(len(batch))
            ]
        )

        return {
            "label": label,
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "batch_idx": batch_idx,
        }

    def prepare_data(self):
        # Load the dataset
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.dataset = afc.data.multisource.load_multisource(self.dataset_path)
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

            columns_to_remove = ["ids", "sentences"]
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
