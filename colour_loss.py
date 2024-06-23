import tensorflow as tf
from tensorflow import keras
import numpy as np

def colour_cross_entropy_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Custom Loss function for colour images for pixelCNN.
    y_true: batch of images (batch_size, image_size, image_size, 3)
    y_pred: probability distribution across all channels (batch_size, image_size, image_size, pixel_levels*3)
    """
    _, _, _, out_channels = y_pred.get_shape()
    total_loss = tf.constant(0.0)
    for i in range(3):
        indices = np.arange(out_channels)
        indices = indices[indices % 3 == i]
        pred_chan = tf.gather(y_pred, indices, axis=-1)
        label_chan = y_true[..., i]
        loss = keras.losses.sparse_categorical_crossentropy(label_chan, pred_chan)
        total_loss = tf.add(total_loss, tf.reduce_mean(loss))
    return total_loss

