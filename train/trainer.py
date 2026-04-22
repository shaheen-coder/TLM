import os
from datetime import datetime
import numpy as np
import tensorflow as tf

from model.transformer import TinyLM
from model.config import ModelConfig

# tensorboard
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print(f"Running on TPU: {resolver.master()}")
except ValueError:
    print("TPU not found; falling back to default strategy (CPU/GPU).")
    strategy = tf.distribute.get_strategy()

# --- Configuration ---
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    write_graph=False,
    update_freq="epoch",
)

# --- Data Loading ---
# Note: Using mmap is fine on TPU VMs as the local disk is shared.
encoder_arr = np.load("datasets/encoder.npy", mmap_mode="r")
decoder_in_arr = np.load("datasets/decoder_in.npy", mmap_mode="r")
decoder_out_arr = np.load("datasets/decoder_out.npy", mmap_mode="r")

dataset = tf.data.Dataset.from_tensor_slices(
    ((encoder_arr, decoder_in_arr), decoder_out_arr)
)
# drop_remainder=True is REQUIRED for TPU performance to keep shapes static
dataset = (
    dataset.shuffle(10000)
    .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

# Validation
val_encoder_arr = np.load("datasets/val_encoder.npy", mmap_mode="r")
val_decoder_in_arr = np.load("datasets/val_decoder_in.npy", mmap_mode="r")
val_decoder_out_arr = np.load("datasets/val_decoder_out.npy", mmap_mode="r")

val_dataset = tf.data.Dataset.from_tensor_slices(
    ((val_encoder_arr, val_decoder_in_arr), val_decoder_out_arr)
)
val_dataset = val_dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True).prefetch(
    tf.data.AUTOTUNE
)


# --- Custom Loss & Metrics (TPU compatible) ---
def masked_pad_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = loss * mask
    # We use global reduction via strategy or manual sum/mean
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)


def masked_accuracy(y_true, y_pred):
    preds = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    match = tf.cast(tf.equal(tf.cast(y_true, tf.int32), preds), tf.float32)
    match *= mask
    return tf.reduce_sum(match) / tf.maximum(tf.reduce_sum(mask), 1.0)


# --- Model Creation within Strategy Scope ---
with strategy.scope():
    config = ModelConfig()
    model = TinyLM(config)

    # Optimizer must also be inside the scope
    optimizer = tf.keras.optimizers.Adam(1e-4)

    model.compile(
        optimizer=optimizer,
        loss=masked_pad_loss,
        metrics=[masked_accuracy],
    )

# --- Training ---
model.fit(
    dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[tb_cb],
)

model.save("tlm.keras")
