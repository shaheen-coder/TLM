from tensorflow.keras.saving import register_keras_serializable
from dataclasses import dataclass


@register_keras_serializable(package="tinylm")
@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 10000
    d_model: int = 128
    num_heads: int = 4
    dff: int = 512
    dropout: float = 0.2
    num_layers: int = 2

    def get_config(self):
        import dataclasses

        return dataclasses.asdict(self)

    @classmethod
    def from_config(cls, config):
        return cls(**config)
