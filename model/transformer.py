from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    Add,
    LayerNormalization,
    Dropout,
)
from tensorflow.keras.saving import (
    register_keras_serializable,
    serialize_keras_object,
    deserialize_keras_object,
)
from keras_hub.layers import RotaryEmbedding
import tensorflow as tf

# custom imoprt
from model.config import ModelConfig
from model.attention import MultiHeadAttention

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
        # --------  ROPE -----------
        self.rope = RotaryEmbedding(
            max_wavelength=10000,
            sequence_axis=1,
            feature_axis=-1,
            name="rope",
        )

        self.dropout = Dropout(config.dropout)

        # ---- Self Attention ----
        self.attention = MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.d_model // config.num_heads,
            dropout=config.dropout,
            name="self_attention",
        )

        self.add1 = Add()
        self.norm1 = LayerNormalization(epsilon=1e-6)

        # ---- Feed Forward ----
        self.ffn = tf.keras.Sequential(
            [
                Dense(config.dff, activation="relu"),
                Dense(config.d_model),
            ]
        )

        self.add2 = Add()
        self.norm2 = LayerNormalization(epsilon=1e-6)

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
        return mask

    # -------- Forward -------- #
    def call(self, inputs, training=False):

        x = self.embedding(inputs)

        # apply rope
        x = self.rope(x)

        x = self.dropout(x, training=training)

        # mask
        attn_mask = self._causal_mask(inputs)

        # ---- Self Attention ----
        attn_out = self.attention(
            query=x,
            key=x,
            value=x,
            attention_mask=attn_mask,
            training=training,
        )
        x = self.add1([x, attn_out])
        x = self.norm1(x)

        # ---- FFN ----
        ffn_out = self.ffn(x, training=training)
        x = self.add2([x, ffn_out])
        x = self.norm2(x)

        # ---- LM head (tied weights) ----
        logits = tf.matmul(x, self.embedding.embeddings, transpose_b=True)
        logits = logits + self.logits_bias

        return logits
