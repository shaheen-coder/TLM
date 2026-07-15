import math
import os
from datetime import datetime
import numpy as np
import tensorflow as tf

from model.transformer import TinyLM
from model.config import ModelConfig
from train.lossbreaker import LossBreaker
# --- Configuration ---
BATCH_SIZE = 64
FT_EPOCH : int = 10
BREAK_LOSS = 1.0

log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
ckpt_dir = os.path.join("checkpoint",datetime.now().strftime("%Y%m%d-%H%M%S"))
nan_cb = tf.keras.callbacks.TerminateOnNaN()

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    write_graph=False,
    update_freq="epoch",
)
ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(ckpt_dir, "best_model.keras"),
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)
early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=3,
    min_delta=1e-4,
    restore_best_weights=True,
    verbose=1,
)
loss_breaker_cb = LossBreaker(train_threshold=BREAK_LOSS, val_threshold=BREAK_LOSS)

# --- Foundation dataset ------
# fd_input_arr = np.load("datasets/pretokens/fd_input.npy", mmap_mode="r")
# fd_target_arr = np.load("datasets/pretokens/fd_target.npy", mmap_mode="r")

# ---- fine tune ds -------
ft_input_arr = np.load("datasets/pretokens/ft_input.npy", mmap_mode="r")
ft_target_arr = np.load("datasets/pretokens/ft_target.npy", mmap_mode="r")

# fd_dataset = tf.data.Dataset.from_tensor_slices((fd_input_arr, fd_target_arr))
# fd_dataset = fd_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


ft_dataset = tf.data.Dataset.from_tensor_slices((ft_input_arr, ft_target_arr))
ft_dataset = ft_dataset.shuffle(10000,reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Validation
val_input_arr = np.load("datasets/pretokens/ft_val_input.npy", mmap_mode="r")
val_target_arr = np.load("datasets/pretokens/ft_val_target.npy", mmap_mode="r")

val_dataset = tf.data.Dataset.from_tensor_slices((val_input_arr, val_target_arr))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Cosine Scheduler --
steps_per_epoch = math.ceil(len(ft_input_arr) // BATCH_SIZE)
total_steps = steps_per_epoch * FT_EPOCH

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=3e-4,
    decay_steps=total_steps,
    alpha=0.1
)

# --- Custom Loss & Metrics ---

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
)

def masked_pad_loss(y_true, y_pred):
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
    learning_rate=lr_schedule, weight_decay=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7
)

model.compile(
    optimizer=optimizer,
    loss=masked_pad_loss,
    metrics=[masked_accuracy],
    jit_compile=True,
    steps_per_execution=32
)

# --- Training ---
# FD_EPOCH : int = 5
# model.fit(
    # fd_dataset,
    # epochs=FD_EPOCH,
    # callbacks=[tb_cb]
# )
model.fit(
    ft_dataset,
    validation_data=val_dataset,
    epochs=FT_EPOCH,
    callbacks=[tb_cb,ckpt_cb, early_stop_cb, nan_cb, loss_breaker_cb],
)

model.save("tlm.keras")
