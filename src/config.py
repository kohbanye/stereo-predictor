from pathlib import Path

from pydantic import BaseModel


class ModelConfig(BaseModel):
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.2


class TrainingConfig(BaseModel):
    batch_size: int = 1024
    epochs: int = 100
    lr: float = 0.001
    num_workers: int = 32
    checkpoints_dir: str = "checkpoints"
    project_name: str = "stereo-predictor"
    run_name: str | None = None


class Config(BaseModel):
    sdf_file: Path
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
