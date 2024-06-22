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
        self.mask = None
    
    def build(self, input_shape: tuple[int]) -> None:
        self.conv_layer.build(input_shape)
        #Kernel shape is KHxKWxDepthxNum Filters
        kernel_shape = self.conv_layer.kernel.get_shape()
        _, _, num_in_channels, num_filters = kernel_shape
        mask = np.zeros(shape=kernel_shape)
        #Initally flip the mask to the shape Num FiltersxDepthxKHxKW to make processing simpler
        mask = np.transpose(mask, axes=(3, 2, 0, 1))

        #Set half - 1 rows to 1.0s
        mask[..., :kernel_shape[0] // 2, :] = 1.0
        #Set the half rows tp 1.0 up to the middle - 1 column
        mask[..., kernel_shape[0] // 2, :kernel_shape[1] // 2] = 1.0
        
        # Adapted from https://github.com/rampage644/wavenet/blob/master/wavenet/models.py
        def bmask(i_out: int, i_in: int) -> np.ndarray:
            cout_idx = np.expand_dims(np.arange(num_filters) % 3 == i_out, 1)
            cin_idx = np.expand_dims(np.arange(num_in_channels) % 3 == i_in, 0)
            a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
            return a1 * a2

        mask[bmask(1, 0), kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0
        mask[bmask(2, 0), kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0
        mask[bmask(2, 1), kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0

        if self.mask_type == "B":
            for i in range(3):
                mask[bmask(i, i), kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0

        mask = np.transpose(mask, axes=(2, 3, 1, 0))

        #A more verbose method for understanding
        # for filter_idx in range(num_filters):
        #     if filter_idx % 3 == 0:
        #         if self.mask_type == "B":
        #             mask_internal = np.arange(num_filters) % 3 == 0
        #             mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ] = 1.0
        #     if filter_idx % 3 == 1:
        #         mask_internal = np.arange(num_filters) % 3 == 0
        #         mask[filter_idx, mask_internal, kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0
        #         if self.mask_type == "B":
        #             mask_internal = np.arange(num_filters) % 3 == 1
        #             mask[filter_idx, mask_internal, kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0
        #     if filter_idx % 3 == 2:
        #         mask_internal = np.arange(num_filters) % 3 == 0
        #         mask[filter_idx, mask_internal, kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0
        #         mask_internal = np.arange(num_filters) % 3 == 1
        #         mask[filter_idx, mask_internal, kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0
        #         if self.mask_type == "B":
        #             mask_internal = np.arange(num_filters) % 3 == 2
        #             mask[filter_idx, mask_internal, kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0
        self.mask = mask
    
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

