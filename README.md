# Pixel CNN

Implements a pixel CNN in Keras.

### Code:
The main code is located in the following files:
* main.py - main entry file for training the network
* main_fashion_mnist.py - same as main.py but for the fashion mnist dataset
* model.py - implements the pixelCNN network
* model_building_blocks.py - residual block and masked conv2D implementations to use in the network
* image_generator.py - Keras callback to plot images whilst training
* display.py - helper function to plot images

We also implement a pixelCNN for multi-channel images in the following files:
* main_colour_images.py
* image_generator_colour.py
* model_colour.py
* model_building_blocks_colour.py
* colour_loss.py

### Other files
* test_custom_loss.py - used to prototype the custom colour loss
