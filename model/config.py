from tensorflow.keras.saving import register_keras_serializable
from dataclasses import dataclass


@register_keras_serializable(package="tinylm")
@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 15000
    d_model: int = 512
    lstm: int = 512
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainerConfig:
    max_seq: int = 45
