import tensorflow as tf
from tensorflow.data import Dataset

from model.transformer import TinyLM
from train.pretokens import PreTokens
from model.config import ModelConfig

dataset = PreTokens("datasets/", "tokenizer/tiny_lm_tokenizer.json")


trainer_dataset = Dataset.from_generator(
    dataset.get_tokens,
    output_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    ),
)

trainer_dataset = trainer_dataset.prefetch(tf.data.AUTOTUNE)

config = ModelConfig()

model = TinyLM(config)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)


model.fit(trainer_dataset, steps_pre_epoch=trainer_dataset.steps, epoch=2)
