from typing import cast

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from src.config import ModelConfig, TrainingConfig


class ChiralityGAT(pl.LightningModule):
    def __init__(
        self,
        num_node_features: int,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = training_config.lr
        self.num_layers = model_config.num_layers
        self.dropout = model_config.dropout
        self.hidden_channels = model_config.hidden_channels
        self.num_layers = model_config.num_layers
        self.classifier_hidden_dim = model_config.classifier_hidden_dim
        self.heads = model_config.heads
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # First GAT layer
        self.convs.append(
            GATConv(
                num_node_features,
                self.hidden_channels,
                heads=self.heads,
                dropout=self.dropout,
            ),
        )
        self.batch_norms.append(torch.nn.BatchNorm1d(self.hidden_channels * self.heads))

        # Hidden GAT layers
        for _ in range(self.num_layers - 2):
            self.convs.append(
                GATConv(
                    self.hidden_channels * self.heads,
                    self.hidden_channels,
                    heads=self.heads,
                    dropout=self.dropout,
                ),
            )
            self.batch_norms.append(
                torch.nn.BatchNorm1d(self.hidden_channels * self.heads)
            )

        # Last GAT layer
        self.convs.append(
            GATConv(
                self.hidden_channels * self.heads,
                self.hidden_channels,
                heads=1,
                dropout=self.dropout,
            ),
        )
        self.batch_norms.append(torch.nn.BatchNorm1d(self.hidden_channels))

        # Output layer for binary classification (R/S)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_channels, self.classifier_hidden_dim),
            torch.nn.BatchNorm1d(self.classifier_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.classifier_hidden_dim, self.classifier_hidden_dim // 2),
            torch.nn.BatchNorm1d(self.classifier_hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.classifier_hidden_dim // 2, 2),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, _, edge_index = (
            cast(torch.Tensor, data.x),
            data.edge_attr,
            data.edge_index,
        )

        prev_x = None
        for i in range(self.num_layers):
            if i > 0 and prev_x is not None and prev_x.size(-1) == x.size(-1):
                # Apply residual connection if dimensions match
                x = x + prev_x
            prev_x = x
            # Apply GAT layer
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)

            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.classifier(x)

        return logits

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        logits = self.forward(batch)
        # Create mask for chiral centers (y != -100)
        mask = batch.y >= 0
        # Apply loss only on chiral centers
        loss = F.cross_entropy(logits[mask], batch.y[mask])

        # Calculate accuracy for logging
        pred = logits[mask].argmax(dim=1)
        acc = (pred == batch.y[mask]).float().mean()

        self.log("train_loss", loss, batch_size=mask.sum())
        self.log("train_acc", acc, batch_size=mask.sum())
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        logits = self.forward(batch)
        # Create mask for chiral centers
        mask = batch.y >= 0
        # Apply loss only on chiral centers
        loss = F.cross_entropy(logits[mask], batch.y[mask])

        # Calculate accuracy
        pred = logits[mask].argmax(dim=1)
        acc = (pred == batch.y[mask]).float().mean()

        self.log("val_loss", loss, batch_size=mask.sum())
        self.log("val_acc", acc, batch_size=mask.sum())
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        logits = self.forward(batch)
        # Create mask for chiral centers
        mask = batch.y >= 0
        # Apply loss only on chiral centers
        loss = F.cross_entropy(logits[mask], batch.y[mask])

        # Calculate accuracy
        pred = logits[mask].argmax(dim=1)
        acc = (pred == batch.y[mask]).float().mean()

        self.log("test_loss", loss, batch_size=mask.sum())
        self.log("test_acc", acc, batch_size=mask.sum())
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
