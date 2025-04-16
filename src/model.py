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
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        # First GAT layer
        self.convs.append(
            GATConv(
                num_node_features, hidden_channels, heads=heads, dropout=self.dropout
            ),
        )

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

        # Last GAT layer
        self.convs.append(
            GATConv(
                hidden_channels * heads, hidden_channels, heads=1, dropout=self.dropout
            ),
        )

        # Output layer for binary classification (R/S)
        self.classifier = torch.nn.Linear(hidden_channels, 2)

    def forward(self, data: Data) -> torch.Tensor:
        x, _, edge_index = (
            cast(torch.Tensor, data.x),
            data.edge_attr,
            data.edge_index,
        )

        # Initial node features
        chiral_indices = cast(torch.Tensor, data.chiral_indices)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Extract features only for chiral centers
        chiral_features = x[chiral_indices]

        # Classification
        x = self.classifier(chiral_features)

        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        out = self.forward(batch)
        chiral_labels = batch.y
        loss = F.nll_loss(out, chiral_labels)
        self.log("train_loss", loss, batch_size=len(chiral_labels))
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        out = self.forward(batch)
        chiral_labels = batch.y
        loss = F.nll_loss(out, chiral_labels)
        pred = out.argmax(dim=1)
        acc = (pred == chiral_labels).float().mean()
        self.log("val_loss", loss, batch_size=len(chiral_labels))
        self.log("val_acc", acc, batch_size=len(chiral_labels))
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        out = self.forward(batch)
        chiral_labels = batch.y
        loss = F.nll_loss(out, chiral_labels)
        pred = out.argmax(dim=1)
        acc = (pred == chiral_labels).float().mean()
        self.log("test_loss", loss, batch_size=len(chiral_labels))
        self.log("test_acc", acc, batch_size=len(chiral_labels))
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
