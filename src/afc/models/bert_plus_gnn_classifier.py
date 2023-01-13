import torch
import pytorch_lightning as pl
import numpy as np
import wandb
from torch import nn
from torch_scatter import scatter_mean
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
from torch.nn import functional as F
from sklearn import metrics
from .gnn_classifier import GNNClassifier
from afc import utils


class BERTPlusGNNClassifier(pl.LightningModule):
    def __init__(
        self,
        num_labels: int = 2,
        model_name: str = "bert-base-cased",
        in_features: int = 768,
        hidden_features: int = 256,
        out_features: int = 2,
        n_heads_gnn: int = 1,
        n_layers: int = 3,
        connectivity: str = "full",
        dropout: float = 0.5,
        learning_rate: float = 0.00002,
        weight_decay: float = 0.01,
        log_f1: bool = False,
        f1_aggr: str = "binary",
        micro_and_macro: bool = False,
        add_positional_encoding: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_loss = []
        self.train_accuracy = []
        self.last_log_step = self.global_step

        self.validation_labels = []
        self.validation_predictions = []
        self.test_labels = []
        self.test_predictions = []


        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_class_weights = None

        self.gnn = GNNClassifier(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=num_labels,
            n_heads=n_heads_gnn,
            n_layers=n_layers,
            pooling="max",
            dropout=dropout,
            learning_rate=learning_rate,
        )

        # Disable BERT finetuning if there should be a warmup period for the GNN
        if self.hparams.warmup_epochs > 0:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def configure_optimizers(self):
        # TODO: Add the posibility for different learning rates of the BERT module and the GNN module
        # https://pytorch.org/docs/1.12/optim.html#per-parameter-options
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer

    def on_train_epoch_start(self):
        # Unfreeze the BERT after the GNN warmup period has passed
        if self.current_epoch >= self.hparams.warmup_epochs:
            for parameter in self.model.parameters():
                parameter.requires_grad = True

    def set_train_class_weights(self, weights):
        self.train_class_weights = weights

    def forward(self, **inputs):

        # forward functions for different models
        # DistilBERT:   input_ids, attention_mask
        # RoBERTa:      input_ids, attention_mask, token_type_ids
        # XLM-RoBERTa:   input_ids, attention_mask, token_type_ids

        # Compute embeddings
        if "token_type_ids" in inputs:
            model_output = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                output_hidden_states=True,
            )
        else:
            model_output = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )

        # Use [CLS] embeddings as the sentences' embeddings
        # embeddings.shape = (part, embedding)
        embeddings = model_output.last_hidden_state[:, 0, :]

        if self.hparams.add_positional_encoding:
            _, counts = torch.unique_consecutive(inputs["batch_idx"], return_counts=True)
            positional_encodings = torch.concat([
                utils.graph.create_positional_encoding(c, embeddings.shape[1], embeddings.device)
                for c in counts
            ], dim=0)

            assert positional_encodings.shape == embeddings.shape
            embeddings = embeddings + positional_encodings

        graph = utils.graph.create_graph_batch_from_nodes(
            x=embeddings,
            y=inputs["label"],
            batch_idx=inputs["batch_idx"],
            connectivity=self.hparams.connectivity,
        )

        # Run the graphs through the GNN
        output = self.gnn(x=graph.x, edge_index=graph.edge_index, batch_idx=graph.batch)

        return output

    def _compute_loss_accuracy(self, probs, predictions, targets, weight=None):
        assert (
            predictions.shape == targets.shape
        ), f"predictions and targets have different shapes, {predictions.shape} and {targets.shape}"

        loss = F.cross_entropy(probs, targets.squeeze(dim=1), reduction="mean", weight=weight)
        accuracy = (predictions == targets).float().mean().item()

        return loss, accuracy

    @staticmethod
    def _compute_micro_macro(y_true, y_pred):
        scores = dict()
        for avg in ["micro", "macro"]:
            scores[avg]["f1"] = metrics.f1_score(y_true, y_pred, average=avg)
            scores[avg]["recall"] = metrics.recall_score(y_true, y_pred, average=avg)
            scores[avg]["precision"] = metrics.precision_score(y_true, y_pred, average=avg)

        return scores

    def _common_step(self, batch, batch_idx, stage):
        assert batch["label"].shape[1] == 1

        outputs = self.forward(**batch)

        predictions = outputs.argmax(dim=1).view(-1, 1)
        loss, accuracy = self._compute_loss_accuracy(
            outputs, predictions, batch["label"],
            weight=self.train_class_weights if stage in ["train", "validation"] else None
        )

        if stage is not None and stage != "train":
            self.log(f"{stage}/loss", loss, batch_size=batch["label"].shape[0])
            self.log(f"{stage}/accuracy", accuracy, batch_size=batch["label"].shape[0])

        return loss, accuracy, predictions

    def training_step(self, batch, batch_idx):
        loss, accuracy, predictions = self._common_step(batch, batch_idx, stage="train")

        # Specific logging of the training metrics
        # Average the metrics since last logged optimization step
        if self.last_log_step != self.global_step:
            self.log("train/loss", np.mean(self.train_loss), on_epoch=True)
            self.log("train/accuracy", np.mean(self.train_accuracy), on_epoch=True)
            self.train_loss = []
            self.train_accuracy = []
            self.last_log_step = self.global_step
        else:
            self.train_loss.append(loss.item())
            self.train_accuracy.append(accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        # if self.trainer.logger and self.trainer.global_step == 0:
        #     wandb.define_metric("validation/accuray", summary="max")
        #     wandb.define_metric("validation/loss", summary="min")

        loss, accuracy, predictions = self._common_step(batch, batch_idx, stage="validation")

        self.validation_labels.append(batch["label"])
        self.validation_predictions.append(predictions)

        return predictions

    def validation_epoch_end(self, validation_step_outputs):
        labels = torch.vstack(self.validation_labels).cpu()
        predictions = torch.vstack(self.validation_predictions).cpu()

        if self.hparams.log_f1:
            if self.hparams.micro_and_macro:
                perf_scores = self._compute_micro_macro(labels, predictions)
                for avg, scores in perf_scores.items():
                    for score, score_val in scores.items():
                        self.log(f"validation/{score}_score.{avg}", score_val)
            else:
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

        self.test_labels.append(batch["label"])
        self.test_predictions.append(predictions)

        return predictions

    def test_epoch_end(self, test_step_outputs):
        if self.hparams.log_f1:
            labels = torch.vstack(self.test_labels).cpu()
            predictions = torch.vstack(self.test_predictions).cpu()

            if self.hparams.micro_and_macro:
                perf_scores = self._compute_micro_macro(labels, predictions)
                for avg, scores in perf_scores.items():
                    for score, score_val in scores.items():
                        self.log(f"test/{score}_score.{avg}", score_val)
            else:
                f1_score = metrics.f1_score(labels, predictions, average=self.hparams.f1_aggr)
                recall_score = metrics.recall_score(labels, predictions, average=self.hparams.f1_aggr)
                precision_score = metrics.precision_score(labels, predictions, average=self.hparams.f1_aggr)

                self.log("test/f1_score", f1_score)
                self.log("test/recall_score", recall_score)
                self.log("test/precision_score", precision_score)

        self.test_labels = []
        self.test_predictions = []

