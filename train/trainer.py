import os
from datetime import datetime
import numpy as np
import tensorflow as tf

from model.transformer import TinyLM
from model.config import ModelConfig

# --- Configuration ---
BATCH_SIZE = 12
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    update_freq="epoch",
)

# --- Foundation dataset ------
fd_input_arr = np.load("datasets/pretokens/fd_input.npy", mmap_mode="r")
fd_target_arr = np.load("datasets/pretokens/fd_target.npy", mmap_mode="r")

# ---- fine tune ds -------
ft_input_arr = np.load("datasets/pretokens/ft_input.npy", mmap_mode="r")
ft_target_arr = np.load("datasets/pretokens/ft_target.npy", mmap_mode="r")

fd_dataset = tf.data.Dataset.from_tensor_slices((fd_input_arr, fd_target_arr))
fd_dataset = fd_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


ft_dataset = tf.data.Dataset.from_tensor_slices((ft_input_arr, ft_target_arr))
ft_dataset = ft_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Validation
val_input_arr = np.load("datasets/pretokens/ft_val_input.npy", mmap_mode="r")
val_target_arr = np.load("datasets/pretokens/ft_val_target.npy", mmap_mode="r")

val_dataset = tf.data.Dataset.from_tensor_slices((val_input_arr, val_target_arr))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# --- Custom Loss & Metrics ---
def masked_pad_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    loss = loss_fn(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = loss * mask

    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)


def masked_accuracy(y_true, y_pred):
    preds = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    match = tf.cast(tf.equal(tf.cast(y_true, tf.int32), preds), tf.float32)

    match *= mask
    return tf.reduce_sum(match) / tf.maximum(tf.reduce_sum(mask), 1.0)


# --- Model Creation (no strategy scope) ---
config = ModelConfig()
model = TinyLM(config)

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-4, weight_decay=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7
)

model.compile(
    optimizer=optimizer,
    loss=masked_pad_loss,
    metrics=[masked_accuracy],
)

# --- Training ---
EPOCH : int = 5
for epoch in range(EPOCH):
    model.fit(
        fd_dataset,
        epochs=1,
        callbacks=[tb_cb],
    )
    model.fit(
        ft_dataset,
        validation_data=val_dataset,
        epochs=1,
        callbacks=[tb_cb],
    )

model.save("tlm.keras")
