import numpy as np
import tensorflow as tf

from model.transformer import TinyLM
from model.config import ModelConfig

encoder_arr = np.load("datasets/encoder.npy", mmap_mode="r")
decoder_in_arr = np.load("datasets/decoder_in.npy", mmap_mode="r")
decoder_out_arr = np.load("datasets/decoder_out.npy", mmap_mode="r")

dataset = tf.data.Dataset.from_tensor_slices(
    ((encoder_arr, decoder_in_arr), decoder_out_arr)
)
dataset = (
    dataset.shuffle(10000).batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
)
config = ModelConfig()

model = TinyLM(config)

# fix loss fn to ingore pad tokens
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def masked_pad_loss(y_true, y_pred):

    loss = loss_fn(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

    loss = loss * mask
    denom = tf.reduce_sum(mask)
    return tf.reduce_sum(loss) / tf.maximum(denom, 1.0)


# kears cant ingore the pad token
def masked_accuracy(y_true, y_pred):
    preds = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

    match = tf.cast(tf.equal(y_true, preds), tf.float32)
    match *= mask

    return tf.reduce_sum(match) / tf.maximum(tf.reduce_sum(mask), 1.0)


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=masked_pad_loss,
    metrics=[masked_accuracy],
)

model.fit(dataset, epochs=50)

model.save("tlm.keras")
