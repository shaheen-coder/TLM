import tensorflow as tf
from tensorflow.data import Dataset

from model.transformer import TinyLM
from train.preprocess import PreTokens
from model.config import ModelConfig

dataset = PreTokens("datasets/", "tokenizer/tiny_lm_tokenizer.json")


def data_generator():
    for encoder_ids, decoder_ids in dataset.get_tokens():
        yield (encoder_ids, decoder_ids[:-1]), decoder_ids[1:]


trainer_dataset = Dataset.from_generator(
    data_generator,
    output_signature=(
        (
            tf.TensorSpec(shape=(None,), dtype=tf.int32),  # encoder_input
            tf.TensorSpec(shape=(None,), dtype=tf.int32),  # decoder_input
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),  # labels
    ),
)

trainer_dataset = trainer_dataset.padded_batch(
    6,
    padded_shapes=(
        ([None], [None]),  # inputs: (encoder, decoder)
        [None],  # labels
    ),
    padding_values=(
        (tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)),
        tf.constant(0, dtype=tf.int32),
    ),
).prefetch(tf.data.AUTOTUNE)

config = ModelConfig()

model = TinyLM(config)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(trainer_dataset, epochs=2)
