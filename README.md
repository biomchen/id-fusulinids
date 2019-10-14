## Experiment on Convolutional Neural Network – Identifying fusulinids                

@author: Meng Chen (under construction)

------------
## Overview

### Data

The original data was provided **Dr. Yikun Shi**, School of Earth Sciences, University. They have been preprocessed.
You can find the examples in the example folder.

### Data augmentation

The original dataset has 119 images, which were far from enough for deep learning neural network. I used `ImageDataGenerator` in `keras.preprocessing.image` to perform the data augmentation.

### Model

The convolutional neural network is implemented with `Keras` API (`TensorFlow` backend).

### Training

The model is trained for 10 epoch with batch generator.
After 10 epochs, the accuracy is about
Loss function for the training is `sparse_categorical_crossentropy`.

---------

## How to use

#### Dependencies

  Tensorflow >=1.14
  Keras == 2.0

#### Run cnn_train.py

This will show how training process.

#### Results

## About `Keras`
