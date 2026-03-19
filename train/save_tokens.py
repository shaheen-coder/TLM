from tensorflow.data import Dataset
import tensorflow as tf
from preprocess import PreTokens


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

trainer_dataset = (
    trainer_dataset.padded_batch(
        batch_size=36,
        padded_shapes=(
            ([None], [None]),  # inputs: (encoder, decoder)
            [None],  # labels
        ),
        padding_values=(
            (tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)),
            tf.constant(0, dtype=tf.int32),
        ),
    )
    .shuffle(10000)
    .prefetch(tf.data.AUTOTUNE)
)

trainer_dataset.save("datasets/tiny_lm_dataset")
