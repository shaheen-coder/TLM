from tensorflow.keras.saving import register_keras_serializable
from dataclasses import dataclass


@register_keras_serializable(package="tinylm")
@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8687
    d_model: int = 256
    lstm: int = 256
    dropout: float = 0.0

    def get_config(self):
        import dataclasses

        return dataclasses.asdict(self)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@dataclass(frozen=True)
class TrainerConfig:
    max_seq: int = 45
