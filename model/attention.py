import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="tinylm")
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, num_heads: int, key_dim: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim          
        self.d_model = num_heads * key_dim
        self.dropout_rate = dropout

        self.wq = tf.keras.layers.Dense(self.d_model, use_bias=False, name="wq")
        self.wk = tf.keras.layers.Dense(self.d_model, use_bias=False, name="wk")
        self.wv = tf.keras.layers.Dense(self.d_model, use_bias=False, name="wv")
        self.out_proj = tf.keras.layers.Dense(self.d_model, name="out_proj")
        self.dropout = tf.keras.layers.Dropout(dropout)

    @staticmethod
    def _phi(x: tf.Tensor) -> tf.Tensor:
        return tf.nn.elu(x) + 1.0

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        batch = tf.shape(x)[0]
        seq   = tf.shape(x)[1]
        x = tf.reshape(x, [batch, seq, self.num_heads, self.key_dim])
        return tf.transpose(x, [0, 2, 1, 3])  

    @staticmethod
    def _apply_mask_to_keys(K: tf.Tensor, V: tf.Tensor, mask: tf.Tensor):

        key_valid = tf.reduce_any(mask, axis=2, keepdims=True)   
        key_valid = tf.cast(key_valid, K.dtype)
        K = K * key_valid  
        V = V * key_valid
        return K, V

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

        Q = self._split_heads(self._phi(self.wq(query)))   
        K = self._split_heads(self._phi(self.wk(key))) 
        V = self._split_heads(self.wv(value))              

        if attention_mask is not None:
            mask = tf.cast(attention_mask, dtype=tf.bool)
            if mask.shape.rank == 3:
                mask = tf.expand_dims(mask, axis=1)
            K, V = self._apply_mask_to_keys(K, V, mask)

        kv = tf.einsum("bhsd,bhse->bhde", K, V)

        k_sum = tf.reduce_sum(K, axis=2)

        denom = tf.einsum("bhnd,bhd->bhn", Q, k_sum) + 1e-6

        out = tf.einsum("bhnd,bhde->bhne", Q, kv)

        out = out / denom[..., tf.newaxis]
        
        out = self.dropout(out, training=training)

        out = tf.transpose(out, [0, 2, 1, 3])
        batch = tf.shape(out)[0]
        seq   = tf.shape(out)[1]
        out = tf.reshape(out, [batch, seq, self.d_model])

        return self.out_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate,
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
