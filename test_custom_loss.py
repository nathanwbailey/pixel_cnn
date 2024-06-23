"""Playground for the custom colour loss."""
import tensorflow as tf
from tensorflow import keras
import numpy as np

PIXEL_LEVELS = 6

y_pred = np.random.rand(3, 16, 16, PIXEL_LEVELS*3)
y_true = np.random.randint(0, 6, size=(3, 16, 16, 3))   

r_pred = y_pred[..., (np.arange(PIXEL_LEVELS*3) % 3 == 0)]
print(r_pred.shape)
print(np.expand_dims(y_true[..., 0], -1).shape)

label = y_true[..., 0].reshape((3, -1))

pred = r_pred.reshape((3, -1, 6))
print(label[0])
print(pred[0])
m = keras.metrics.SparseCategoricalCrossentropy()
loss = keras.losses.sparse_categorical_crossentropy(label, pred)
print(tf.reduce_mean(loss))




y_pred = tf.random.uniform(shape=(3, 16, 16, PIXEL_LEVELS*3))
y_true = tf.random.uniform(shape=(3, 16, 16, 3), minval=0, maxval=6, dtype=tf.int32)
indices = np.arange(PIXEL_LEVELS * 3)
indices = indices[indices % 3 == 0]
r_pred = tf.gather(y_pred, indices, axis=-1)


loss = keras.losses.sparse_categorical_crossentropy(y_true[..., 0], r_pred)
print(tf.reduce_mean(loss))

label = tf.reshape(y_true[..., 0], (3, -1))
pred = tf.reshape(r_pred, (3, -1, 6))
loss = keras.losses.sparse_categorical_crossentropy(label, pred)
print(tf.reduce_mean(loss))

# print(r_pred.get_shape())