import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from keras_hub.layers import RotaryEmbedding
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="tinylm")
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        dropout: float = 0.0,
        rope_max_wavelength: int = 10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.d_model = num_heads * key_dim
        self.dropout_rate = dropout
        self.rope_max_wavelength = rope_max_wavelength

        self.wq = Dense(self.d_model, use_bias=False, name="wq")
        self.wk = Dense(self.d_model, use_bias=False, name="wk")
        self.wv = Dense(self.d_model, use_bias=False, name="wv")
        self.out_proj = Dense(self.d_model, name="out_proj")
        self.dropout = Dropout(dropout)

        self.rope = RotaryEmbedding(
            max_wavelength=rope_max_wavelength,
            sequence_axis=1,
            feature_axis=-1,
            name="rope",
        )

        self.scale = tf.cast(self.key_dim ** -0.5, tf.float32)

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        batch = tf.shape(x)[0]
        seq = tf.shape(x)[1]
        x = tf.reshape(x, [batch, seq, self.num_heads, self.key_dim])
        return x

    def _merge_heads(self, x: tf.Tensor) -> tf.Tensor:
        batch = tf.shape(x)[0]
        seq = tf.shape(x)[1]
        x = tf.reshape(x, [batch, seq, self.d_model])
        return x

    @staticmethod
    def _expand_attention_mask(attention_mask: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(attention_mask, tf.bool)
        if mask.shape.rank == 3:
            mask = mask[:, tf.newaxis, :, :]
        elif mask.shape.rank != 4:
            raise ValueError(
                "attention_mask must have rank 3 or 4 "
                f"(got rank={mask.shape.rank})."
            )
        return mask

    def call(
        self,
        query: tf.Tensor,
        value: tf.Tensor,
        key: tf.Tensor = None,
        attention_mask=None,
        training: bool = False,
    ) -> tf.Tensor:
        if key is None:
            key = value

        q = self._split_heads(self.wq(query))  
        k = self._split_heads(self.wk(key)) 
        v = self._split_heads(self.wv(value))  

        # RoPE 
        q = self.rope(q)
        k = self.rope(k)

        # B, H, T, Dh)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores * tf.cast(self.scale, scores.dtype)

        if attention_mask is not None:
            mask = self._expand_attention_mask(attention_mask)
            neg_inf = tf.cast(-1e9, scores.dtype)
            scores = tf.where(mask, scores, neg_inf)

        attn = tf.nn.softmax(scores, axis=-1)
        attn = self.dropout(attn, training=training)

        out = tf.matmul(attn, v)  # (B, H, T, Dh)
        out = tf.transpose(out, [0, 2, 1, 3])  # (B, T, H, Dh)
        out = self._merge_heads(out)           # (B, T, D)

        return self.out_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "dropout": self.dropout_rate,
                "rope_max_wavelength": self.rope_max_wavelength,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
