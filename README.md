# Autoencoders

Autoencoders are a type of deep learning model used for unsupervised learning. The key layers of autoencoders are the input layer, encoder, bottleneck (hidden layer), decoder, and output layer.

The three main layers of the autoencoder are:

- **Encoder**: Compresses the input data into an encoded representation, which is typically much smaller than the input data.
- **Latent Space Representation / Bottleneck / Code**: This is a compact summary of the input, containing the most important features.
- **Decoder**: Decompresses the encoded representation and reconstructs the data back from its compressed form. A loss function is used at the top to compare the input and output. _Note: It is required that the dimensionality of the input and output be the same. The internal layers (encoder, bottleneck, decoder) can be adjusted._

Autoencoders have a wide variety of real-world applications. Some popular examples include:

- **Transformers and Big Bird**: Autoencoders are components of these algorithms, used for tasks like text summarization and text generation.
- **Image Compression**: Reducing the size of images while maintaining important features.
- **Nonlinear version of PCA**: Autoencoders can be used as a more flexible, nonlinear alternative to Principal Component Analysis (PCA).

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
- Conv2D → ReLU → MaxPooling2D - Relu to battle against the vanishing gradient problem
- Conv2D → ReLU → MaxPooling2D

### Decoder
- Conv2DTranspose → ReLU → Upsampling2D
- Conv2DTranspose → Sigmoid

