import tensorflow as tf 


class LossBreaker(tf.keras.callbacks.Callback):
    def __init__(self, train_threshold=1.0, val_threshold=1.0):
        super().__init__()
        self.train_threshold = train_threshold
        self.val_threshold = val_threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        # Stop on bad numerical behavior
        if loss is not None and (tf.math.is_nan(loss) or tf.math.is_inf(loss)):
            print(f"\nStopping: train loss became invalid at epoch {epoch + 1}")
            self.model.stop_training = True
            return

        if val_loss is not None and (tf.math.is_nan(val_loss) or tf.math.is_inf(val_loss)):
            print(f"\nStopping: val loss became invalid at epoch {epoch + 1}")
            self.model.stop_training = True
            return

        # Stop if loss gets too low for your rule
        if loss is not None and loss <= self.train_threshold:
            print(f"\nStopping: train loss reached {loss:.6f} (<= {self.train_threshold})")
            self.model.stop_training = True
            return

        if val_loss is not None and val_loss <= self.val_threshold:
            print(f"\nStopping: val loss reached {val_loss:.6f} (<= {self.val_threshold})")
            self.model.stop_training = True
            return
