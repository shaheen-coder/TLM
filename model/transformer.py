from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention
from tensorflow.keras.saving import (
    register_keras_serializable,
    serialize_keras_object,
    deserialize_keras_object,
)
import tensorflow as tf

# custom imoprt
from model.config import ModelConfig

from typing import Tuple


@register_keras_serializable(package="tinylm")
class TinyLM(Model):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        # shared embedding
        self.embedding = Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            mask_zero=True,
            embeddings_initializer="glorot_uniform",
            name="shared_embedding",
        )
        # -------------  encoder -------------
        self.encoder_lstm = LSTM(
            units=config.d_model,
            return_state=True,
            return_sequences=True,
            dropout=config.dropout,
            name="encoder_lstm",
        )
        # ------------- Decoder -------------
        self.decoder_lstm = LSTM(
            units=config.d_model,
            return_state=True,
            return_sequences=True,
            dropout=config.dropout,
            name="decoder_lstm",
        )
        # --------------- Attention --------
        self.attention = Attention(
            use_scale=False,
            name="luong_attention",
        )
        # -------------- Attention proj ----
        self.atn_proj = Dense(
            units=config.d_model, activation="tanh", name="attn_projection"
        )

        # ------------- FF -------------
        self.logits_bias = self.add_weight(
            shape=(config.vocab_size,),
            initializer="zeros",
            trainable=True,
            name="logits_bias",
        )

    def get_config(self):
        config = super().get_config()
        config.update({"config": serialize_keras_object(self.config)})
        return config

    @classmethod
    def from_config(cls, config):
        config_dict = config.pop("config")
        model_config = deserialize_keras_object(config_dict)
        return cls(config=model_config, **config)

    def encoder(
        self, encoder_input: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, list[tf.Tensor]]:
        x = self.embedding(encoder_input)

        mask = self.embedding.compute_mask(encoder_input)

        encoder_outputs, h, c = self.encoder_lstm(x, mask=mask, training=training)

        return encoder_outputs, [h, c]

    def decoder(
        self,
        encoder_outputs: tf.Tensor,
        decoder_inputs: tf.Tensor,
        states,
        training: bool = False,
    ):
        y = self.embedding(decoder_inputs)

        mask = self.embedding.compute_mask(decoder_inputs)

        out, h, c = self.decoder_lstm(
            y, mask=mask, initial_state=states, training=training
        )
        context = self.attention([out, encoder_outputs], training=training)
        combined = tf.concat([context, out], axis=-1)
        attn_hidden = self.atn_proj(combined)

        logits = (
            attn_hidden @ tf.transpose(self.embedding.embeddings) + self.logits_bias
        )

        return logits, [h, c]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:

        encoder_input = inputs[0]
        decoder_input = inputs[1]
        # ------------- Encoder Part -------------
        encoder_outputs, encoder_state = self.encoder(encoder_input, training=training)
        # ------------- Decoder Part -------------
        logits, _ = self.decoder(
            encoder_outputs, decoder_input, encoder_state, training=training
        )

        return logits
