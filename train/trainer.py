import tensorflow as tf
from tensorflow.data import Dataset

from model.transformer import TinyLM
from train.preprocess import PreTokens
from model.config import ModelConfig

trainer_dataset = Dataset.load("datasets/tiny_lm_dataset")

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
    metrics=masked_accuracy,
)

model.fit(trainer_dataset, epochs=5)

model.save("tlm.keras")
