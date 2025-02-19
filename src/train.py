import argparse
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import Config
from src.data import ChiralityDataModule
from src.model import ChiralityGAT


def main() -> None:
    parser = argparse.ArgumentParser(description="Train chirality prediction model")
    parser.add_argument("--sdf_file", type=str, required=True, help="Path to SDF file")
    args = parser.parse_args()

    # Initialize config
    config = Config(sdf_file=Path(args.sdf_file))

    # Initialize data module
    datamodule = ChiralityDataModule(
        sdf_file=str(config.sdf_file),
        batch_size=config.training.batch_size,
    )

    # Initialize model
    model = ChiralityGAT(
        num_node_features=datamodule.num_node_features,
        hidden_channels=config.model.hidden_channels,
        num_layers=config.model.num_layers,
        learning_rate=config.training.lr,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.training.checkpoints_dir,
        filename="best-model",
        save_top_k=1,
        monitor="val_acc",
        mode="max",
    )
    wandb_logger = WandbLogger(
        project=config.training.project_name,
        name=config.training.run_name,
        log_model=True,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
    )

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Test the model
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
