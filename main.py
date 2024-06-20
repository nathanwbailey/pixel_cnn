from model import build_model
from image_generator import ImageGenerator
from display import display
import tensorflow as tf
import numpy as np
from tensorflow import keras

IMAGE_SIZE = 16
PIXEL_LEVELS = 32
N_FILTERS = 128
RESIDUAL_BLOCKS = 5
BATCH_SIZE = 128
EPOCHS = 450

(x_train, _), (_, _) = keras.datasets.mnist.load_data()

def preprocess_images(images: np.ndarray) -> tuple[np.ndarray]:
    imgs_int = np.expand_dims(images, -1)
    imgs_int = tf.image.resize(imgs_int, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
    imgs_int = (imgs_int / (256 / PIXEL_LEVELS)).astype(np.uint8)
    imgs = imgs_int.astype("float32")
    imgs = imgs / PIXEL_LEVELS
    return imgs, imgs_int

input_data, output_data = preprocess_images(x_train)

model = build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), num_residual_blocks=5, num_filters=N_FILTERS, pixel_levels=PIXEL_LEVELS)

model.summary()
opt = keras.optimizers.Adam(learning_rate=0.01)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1, min_lr=1e-7, min_delta=1e-4)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=1, patience=10)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy")

img_generator_callback = ImageGenerator(num_img=10, pixel_levels=PIXEL_LEVELS)
model.fit(input_data, output_data, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[lr_scheduler, early_stopping, img_generator_callback], verbose='auto')

generated_images = img_generator_callback.generate(temperature=1.0)
display(generated_images, save_to="./output/gen_images.png")

