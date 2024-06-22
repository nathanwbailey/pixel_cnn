from tensorflow import keras
from model_building_blocks_colour import ResidualBlock
from model_building_blocks_colour import MaskedConv2D

def build_model(input_shape: tuple[int], num_residual_blocks: int, num_filters: int, pixel_levels: int) -> keras.models.Model:
    input = keras.layers.Input(shape=input_shape)
    x = MaskedConv2D(mask_type='A', filters=num_filters, kernel_size=7, activation="relu", padding="same")(input)

    for _ in range(num_residual_blocks):
        x = ResidualBlock(filters=num_filters)(x)
    
    for _ in range(2):
        x = MaskedConv2D(mask_type="B", filters=num_filters, kernel_size=1, padding="valid", activation="relu")(x)
    
    output = keras.layers.Conv2D(filters=pixel_levels*input_shape[-1], kernel_size=1, activation="softmax", padding="valid")(x)
    return keras.models.Model(input, output)
