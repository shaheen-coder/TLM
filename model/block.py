from tensorflow.keras.saving import register_keras_serializable
from model.attention import MultiHeadAttention
from tensorflow.keras.layers import Dense, Dropout, Add, LayerNormalization
import tensorflow as tf

@register_keras_serializable(package="tinylm")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout: float = 0.0,
        rope_max_wavelength: int = 10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout
        self.rope_max_wavelength = rope_max_wavelength

        key_dim = d_model // num_heads

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
            rope_max_wavelength=rope_max_wavelength,
            name="self_attention",
        )
        self.add1 = Add()
        self.norm1 = LayerNormalization(epsilon=1e-6, name="attn_norm")

        self.ffn = tf.keras.Sequential(
            [
                Dense(dff, activation="relu"),
                Dense(d_model),
            ],
            name="ffn",
        )
        self.add2 = Add()
        self.norm2 = LayerNormalization(epsilon=1e-6, name="ffn_norm")
        self.dropout = Dropout(dropout)

    def call(self, x, attention_mask=None, padding_mask=None, training=False):
        attn_out = self.attention(
            query=x,
            key=x,
            value=x,
            attention_mask=attention_mask,
            training=training,
        )
        attn_out = self.dropout(attn_out, training=training)
        x = self.add1([x, attn_out])
        x = self.norm1(x)

        if padding_mask is not None:
            x *= tf.cast(padding_mask[..., tf.newaxis], x.dtype)

        ffn_out = self.ffn(x, training=training)
        ffn_out = self.dropout(ffn_out, training=training)
        x = self.add2([x, ffn_out])
        x = self.norm2(x)

        if padding_mask is not None:
            x *= tf.cast(padding_mask[..., tf.newaxis], x.dtype)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "dropout": self.dropout_rate,
                "rope_max_wavelength": self.rope_max_wavelength,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

