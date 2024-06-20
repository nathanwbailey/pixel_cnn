import tensorflow as tf
import numpy as np
from tensorflow import keras
from display import display

class ImageGenerator(keras.callbacks.Callback):
    def __init__(self, num_img: int, pixel_levels: int) -> None:
        self.num_img = num_img
        self.pixel_levels = pixel_levels
    
    def sample_from(self, probs: tf.Tensor, temperature: float) -> np.ndarray:
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        #Choose a random number using the distribution
        return np.random.choice(len(probs), p=probs)

    def generate(self, temperature: float) -> np.ndarray:
        generated_images = np.zeros(shape=(self.num_img,) + self.model.input_shape[1:])
        _, rows, cols, channels = generated_images.shape
        for row in range(rows):
            for col in range(cols):
                for channel in range(channels):
                    #Predict pixels one by one using the previously predicted pixels
                    probs = self.model.predict(generated_images, verbose=0)[:, row, col, :]
                    generated_images[:, row, col, channel] = [
                        self.sample_from(x, temperature) for x in probs
                    ]
                    generated_images[:, row, col, channel] /= self.pixel_levels
        return generated_images
    
    def on_epoch_end(self, epoch, logs=None) -> None:
        generated_images = self.generate(temperature=1.0)
        display(
            generated_images,
            save_to=f"./output/generated_img_{epoch}.png",
        )
