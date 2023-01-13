import torch
import torch.nn as nn
import torch_geometric as geom
import torch_geometric.nn as geom_nn
import pytorch_lightning as pl
import wandb
from torch.nn import functional as F


class GNNClassifier(pl.LightningModule):
    def __init__(
        self,
        in_features: int = 768,
        hidden_features: int = 128,
        out_features: int = 2,
        n_heads: int = 2,
        n_layers: int = 2,
        share_weights: bool = False,
        pooling: str = "max",
        dropout: float = 0.5,
        learning_rate: float = 0.00002,
    ):
        super().__init__()

        assert n_layers > 0, "At least one GNN layer is required"
        assert n_heads > 0, "At least one head is required"

        # Save the init arguments into self.hparams, comes from the LightningModule
        self.save_hyperparameters()

        gnn_layers = [
            geom_nn.GATv2Conv(
                in_channels=self.hparams.in_features,
                out_channels=self.hparams.hidden_features,
                heads=self.hparams.n_heads,
                concat=True,
                bias=True,
            )
        ]

        if self.hparams.n_heads > 1:
            gnn_layers.append(
                nn.Linear(
                    self.hparams.n_heads * self.hparams.hidden_features,
                    self.hparams.hidden_features,
                )
            )

        for idx in range(1, self.hparams.n_layers):
            gnn_layers.append(nn.ReLU())
            gnn_layers.append(nn.Dropout(p=self.hparams.dropout))

            if self.hparams.share_weights:
                gnn_layers.append(gnn_layers[0])
            else:
                gnn_layers.append(
                    geom_nn.GATv2Conv(
                        in_channels=self.hparams.hidden_features,
                        out_channels=self.hparams.hidden_features,
                        heads=self.hparams.n_heads,
                        concat=True,
                        bias=True,
                    )
                )

            if self.hparams.n_heads > 1:
                if self.hparams.share_weights:
                    gnn_layers.append(gnn_layers[1])
                else:
                    gnn_layers.append(
                        nn.Linear(
                            self.hparams.n_heads * self.hparams.hidden_features,
                            self.hparams.hidden_features,
                        )
                    )

        gnn_layers.append(nn.ReLU())

        self.layers = nn.ModuleList(gnn_layers)

        if pooling == "mean":
            self.pooling = geom_nn.global_mean_pool
        elif pooling == "max":
            self.pooling = geom_nn.global_max_pool
        else:
            self.pooling = geom_nn.global_max_pool

        self.linear = nn.Linear(
            in_features=self.hparams.hidden_features,
            out_features=self.hparams.out_features
        )

    @staticmethod
    def add_model_args(parent_parser, prefix: str = ""):
        parser = parent_parser.add_argument_group("GNNClassifier")
        parser.add_argument(f"--{prefix}in-features", default=768, type=int)
        parser.add_argument(f"--{prefix}hiddent-features", default=512, type=int)
        parser.add_argument(f"--{prefix}out-features", default=2, type=int)
        parser.add_argument(f"--{prefix}n-heads", default=1, type=int)
        parser.add_argument(f"--{prefix}n-layers", default=3, type=int)
        parser.add_argument(f"--{prefix}pooling", default="mean", type=str)

        return parent_parser

    def forward(self, x, edge_index, batch_idx):
        x = x.float()

        for idx, layer in enumerate(self.layers):
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)

        x = self.pooling(x, batch_idx)
        x = self.linear(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def _compute_loss_accuracy(self, y_hat, predictions, y):
        assert (
            y_hat.shape[0] == y.shape[0]
        ), f"predicted probabilities y_hat have different shape than the target y, {y_hat.shape} and {y.shape}"
        assert (
            predictions.shape == y.shape
        ), f"predictions have different shape the the target y, {predictions.shape} and {y.shape}"
        loss = F.cross_entropy(y_hat, y, reduction="mean")
        accuracy = (predictions == y).float().mean()

        return loss, accuracy

    def _common_step(self, batch, batch_idx, stage="train"):
        y_hat = self.forward(batch.x, batch.edge_index, batch.batch)
        predictions = y_hat.argmax(dim=1)

        loss, acc = self._compute_loss_accuracy(y_hat, predictions, batch.y)

        self.log(
            f"{stage}/loss", loss.item(), batch_size=batch.y.shape[0], on_epoch=True
        )
        self.log(f"{stage}/accuracy", acc, batch_size=batch.y.shape[0], on_epoch=True)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx, stage="train")

        return loss

    def validation_step(self, batch, batch_idx):
        if self.trainer.logger and self.trainer.global_step == 0:
            wandb.define_metric("validation/accuracy", summary="max")

        loss, acc = self._common_step(batch, batch_idx, stage="validation")

    def test_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx, stage="test")
