from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 15000
    d_model: int = 512
    lstm: int = 512
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainerConfig:
    max_seq: int = 45
