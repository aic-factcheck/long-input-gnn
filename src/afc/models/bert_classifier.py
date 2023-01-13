import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn import metrics
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset


class BERTClassifier(pl.LightningModule):
    def __init__(
        self,
        num_labels: int = 2,
        learning_rate: float = 0.00001,
        weight_decay: float = 0.01,
        model_name: str = "bert-base-cased",
        log_f1: bool = False,
        f1_aggr: str = "binary",
    ):
        super().__init__()

        self.save_hyperparameters()

        self.validation_labels = []
        self.validation_predictions = []
        self.test_labels = []
        self.test_predictions = []

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer

    def forward(self, **inputs):
        return self.model(**inputs)

    def set_train_class_weights(self, weights):
        self.train_class_weights = weights

    def _compute_accuracy(self, predictions, targets):
        assert (
            predictions.shape == targets.shape
        ), "predictions and targets have different shapes"

        return (predictions == targets).float().mean().item()

    def _common_step(self, batch, batch_idx, stage):

        if self.hparams.model_name == "allenai/longformer-base-4096":
            # set global attention for the CLS token
            batch["attention_mask"][:, 0] = 2

        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            labels=batch["label"],
        )

        predictions = outputs.logits.argmax(dim=1).view(-1, 1)
        accuracy = self._compute_accuracy(predictions, batch["label"])

        if self.train_class_weights is not None:
            loss = F.cross_entropy(
                outputs.logits,
                batch["label"].squeeze(dim=1),
                reduction="mean", weight=self.train_class_weights
            )
        else:
            loss = outputs.loss

        self.log(f"{stage}/loss", loss, on_epoch=True)
        self.log(f"{stage}/accuracy", accuracy, on_epoch=True)

        return loss, self._compute_accuracy(predictions, batch["label"]), predictions

    def training_step(self, batch, batch_idx):
        loss, accuracy, _ = self._common_step(batch, batch_idx, stage="train")

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, predictions = self._common_step(batch, batch_idx, stage="validation")

        if self.hparams.log_f1:
            self.validation_labels.append(batch["label"])
            self.validation_predictions.append(predictions)

        return predictions

    def validation_epoch_end(self, validation_step_outputs):
        if self.hparams.log_f1:
            labels = torch.vstack(self.validation_labels).cpu()
            predictions = torch.vstack(self.validation_predictions).cpu()

            f1_score = metrics.f1_score(labels, predictions, average=self.hparams.f1_aggr)
            recall_score = metrics.recall_score(labels, predictions, average=self.hparams.f1_aggr)
            precision_score = metrics.precision_score(labels, predictions, average=self.hparams.f1_aggr)

            self.log("validation/f1_score", f1_score)
            self.log("validation/recall_score", recall_score)
            self.log("validation/precision_score", precision_score)

        self.validation_labels = []
        self.validation_predictions = []

    def test_step(self, batch, batch_idx):
        loss, accuracy, predictions = self._common_step(batch, batch_idx, stage="test")

        if self.hparams.log_f1:
            self.test_labels.append(batch["label"])
            self.test_predictions.append(predictions)

        return predictions

    def test_epoch_end(self, test_step_outputs):
        if self.hparams.log_f1:
            labels = torch.vstack(self.test_labels).cpu()
            predictions = torch.vstack(self.test_predictions).cpu()

            f1_score = metrics.f1_score(labels, predictions, average=self.hparams.f1_aggr)
            recall_score = metrics.recall_score(labels, predictions, average=self.hparams.f1_aggr)
            precision_score = metrics.precision_score(labels, predictions, average=self.hparams.f1_aggr)

            self.log("test/f1_score", f1_score)
            self.log("test/recall_score", recall_score)
            self.log("test/precision_score", precision_score)

        self.test_labels = []
        self.test_predictions = []

