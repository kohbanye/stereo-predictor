from typing import cast

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class ChiralityGAT(pl.LightningModule):
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        classifier_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # First GAT layer
        self.convs.append(
            GATConv(
                num_node_features, hidden_channels, heads=heads, dropout=self.dropout
            ),
        )
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels * heads))

        # Hidden GAT layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=self.dropout,
                ),
            )
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels * heads))

        # Last GAT layer
        self.convs.append(
            GATConv(
                hidden_channels * heads, hidden_channels, heads=1, dropout=self.dropout
            ),
        )
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        # Output layer for binary classification (R/S)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, classifier_hidden_dim),
            torch.nn.BatchNorm1d(classifier_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            torch.nn.BatchNorm1d(classifier_hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(classifier_hidden_dim // 2, 2),
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
