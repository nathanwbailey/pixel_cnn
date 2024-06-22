import tensorflow as tf
import numpy as np
from tensorflow import keras
from typing import Literal
from typing import Any


class MaskedConv2D(keras.layers.Layer):
    def __init__(self, mask_type: Literal["A", "B"], **kwargs: Any) -> None:
        super().__init__()
        self.mask_type = mask_type
        self.conv_layer = keras.layers.Conv2D(**kwargs)
    
    def build(self, input_shape: tuple[int]) -> None:
        self.conv_layer.build(input_shape)
        #Kernel shape is F1xF2xDepthxNum Filters
        kernel_shape = self.conv_layer.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        #Set half - 1 rows to 1.0s
        self.mask[:kernel_shape[0] // 2, ...] = 1.0
        #Set the half rows tp 1.0 up to the middle - 1 column
        self.mask[kernel_shape[0] // 2, :kernel_shape[1] // 2, ...] = 1.0
        #If the mask type is B, set the middle value to 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0
    
    def call(self, input: tf.Tensor) -> tf.Tensor:
        self.conv_layer.kernel.assign(self.conv_layer.kernel*self.mask)
        return self.conv_layer(input)



class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv_layer_1 = keras.layers.Conv2D(filters = filters, kernel_size=1, activation="relu")
        self.pixel_conv = MaskedConv2D(mask_type="B", filters = filters // 2, kernel_size=3, activation="relu", padding="same")
        self.conv_layer_2 = keras.layers.Conv2D(filters=filters, kernel_size=1, activation="relu")
    
    def call(self, input: tf.Tensor) -> tf.Tensor:
        """
        Spatial or depth dimensions will not change between input and output.
        So no need to downsample anything, we just add input and output.
        """
        x = self.conv_layer_1(input)
        x = self.pixel_conv(x)
        x = self.conv_layer_2(x)
        return (input + x)

