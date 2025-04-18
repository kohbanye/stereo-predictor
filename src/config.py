from pathlib import Path

from pydantic import BaseModel


class ModelConfig(BaseModel):
    hidden_channels: int = 64
    num_layers: int = 3
    classifier_hidden_dim: int = 128
    dropout: float = 0.2


class TrainingConfig(BaseModel):
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-4
    num_workers: int = 8
    checkpoints_dir: str = "checkpoints"
    project_name: str = "stereo-predictor"
    run_name: str | None = None


class Config(BaseModel):
    sdf_file: Path
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
