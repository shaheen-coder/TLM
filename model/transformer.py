from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Embedding,
    Dropout,
)
from tensorflow.keras.saving import (
    register_keras_serializable,
    serialize_keras_object,
    deserialize_keras_object,
)
import tensorflow as tf

# custom imoprt
from model.config import ModelConfig
from model.block import TransformerBlock
# from model.attention import MultiHeadAttention

@register_keras_serializable(package="tinylm")
class TinyLM(Model):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        # ---- Embedding ----
        self.embedding = Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            mask_zero=True,
            name="token_embedding",
        )
        self.dropout = Dropout(config.dropout)

        # ---- Self Attention ----
        self.blocks = [
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                dff=config.dff,
                dropout=config.dropout,
                rope_max_wavelength=getattr(config, "rope_max_wavelength", 10000),
                name=f"block_{i}",
            )
            for i in range(config.num_layers)
        ]
        
        # ---- LM head bias ----
        self.logits_bias = self.add_weight(
            shape=(config.vocab_size,),
            initializer="zeros",
            trainable=True,
            name="logits_bias",
        )
         

    # -------- Serialization -------- #
    def get_config(self):
        config = super().get_config()
        config.update({"config": serialize_keras_object(self.config)})
        return config

    @classmethod
    def from_config(cls, config):
        config_dict = config.pop("config")
        model_config = deserialize_keras_object(config_dict)
        return cls(config=model_config, **config)

    # -------- Mask -------- #
    def _causal_mask(self, x):
        seq_len = tf.shape(x)[1]
        padding_mask = self.embedding.compute_mask(x)
        padding_mask = tf.cast(padding_mask, tf.bool)
        causal_mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0
        )

        mask = padding_mask[:, tf.newaxis, :] & causal_mask[tf.newaxis, :, :]
        return mask,padding_mask

    # -------- Forward -------- #
    def call(self, inputs, training=False):

        x = self.embedding(inputs)

        x = self.dropout(x, training=training)

        # mask
        attn_mask,padding_mask = self._causal_mask(inputs)

        # Keep padded positions quiet throughout the network.
        x *= tf.cast(padding_mask[..., tf.newaxis], x.dtype)

        for block in self.blocks:
            x = block(
                    x,
                    attention_mask=attn_mask,
                    padding_mask=padding_mask,
                    training=training,
                )
            
        # ---- LM head (tied weights) ----
        logits = tf.matmul(x, self.embedding.embeddings, transpose_b=True)
        logits = logits + self.logits_bias

        return logits
