## Experiment on Convolutional Neural Network – Identifying fusulinids

(under construction)               

@author: Meng Chen

------------
## Overview

### Data

The original data was provided by [Dr. Yikun Shi](https://es.nju.edu.cn/crebee/fjs/list.htm), Deputy Director of the Centre for Research and Education on Biological Evolution and Environment, Nanjing University. They have been preprocessed.
You can find the examples in the example folder.

### Data augmentation

The original dataset has 119 images, which were far from enough for deep learning neural network. I used `ImageDataGenerator` in `keras.preprocessing.image` to perform the data augmentation.

### Model

The convolutional neural network (CNN) is implemented with `Keras` API (`TensorFlow` backend).

### Training

The model is trained for 10 epoch with batch generator.
After 10 epochs, the accuracy is about
Loss function for the training is `sparse_categorical_crossentropy`.

---------

## How to use

#### Dependencies

  Tensorflow >=1.14                          
  Keras == 2.0
#### Run eigenface.py

This will calculate the eigenface of all fusulinids in the example folder.

#### Run cnn_train.py

This will show how training process.

#### Results

## About `CNN`
