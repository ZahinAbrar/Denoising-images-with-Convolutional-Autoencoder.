# Image Denoising with Convolutional Autoencoder

This project demonstrates how to build a **Convolutional Autoencoder** (CAE) to denoise images. Autoencoders are neural networks that learn to compress data (encoding) and then reconstruct it (decoding). The convolutional variant leverages the spatial structure of images, making it ideal for image-related tasks.

## Problem Description

We aim to remove noise from images by training a convolutional autoencoder. The model will take noisy images as input and learn to output the clean, original images.

## Dataset

For this project, we use a dataset of clean images and add random Gaussian noise to Custom image datasets

## Model Architecture

The autoencoder consists of two main parts:

1. **Encoder**: This part compresses the image into a lower-dimensional representation by applying a series of convolutional and pooling layers.
2. **Decoder**: The decoder reconstructs the image by applying upsampling and transposed convolutional layers to revert the compressed representation to the original image size.

### Encoder
- Conv2D → ReLU → MaxPooling2D
- Conv2D → ReLU → MaxPooling2D

### Decoder
- Conv2DTranspose → ReLU → Upsampling2D
- Conv2DTranspose → Sigmoid

