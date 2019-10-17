#!/usr/bin/env python

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import os
import glob
import numpy as np
import pathlib
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Conv2D, Reshape
from keras.layers import MaxPooling2D, Activation, Dropout
from keras import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


def generator(b_size, img_height, img_width, dtype, class_mode='sparse'):
    image_generator = ImageDataGenerator(rescale=1./255)
    path = input('Enter your {} data directory...'.format(dtype))
    data_dir= pathlib.Path(path)
    #image_count = len(list(data_dir_train.glob('*/*.jpg')))
    items = data_dir.glob('*')
    genus_names = np.array([item.name for item in items])
    gen = image_generator.flow_from_directory(directory=str(data_dir),
                                              batch_size=b_size,
                                              shuffle=True,
                                              target_size=(img_height,
                                                           img_width),
                                              class_mode=class_mode,
                                              classes=list(genus_names))
    return gen


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        plt.title(genus_names[np.int(label_batch[n])])
        plt.axis('off')

def build_cnn(input_size=(255, 255, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(128, 3, activation='relu')(inputs)
    conv1 = Conv2D(128, 3, activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(4, 4))(conv1)
    flat1 = Flatten()(pool1)
    relu1 = Activation('relu')(flat1)
    drop1 = Dropout(rate=0.5)(relu1)
    dense1 = Dense(64, activation='relu')(drop1)
    dense2 = Dense(6, activation='softmax')(dense1)
    model = Model(inputs=inputs, outputs=dense2)
    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history):
    '''plot the history of the cnn model'''
    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)
    figs, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(epochs, loss, 'bo', label='Training loss')
    axes[0].plot(epochs, val_loss, 'b', label='Validation loss')
    axes[0].set_title('Training and validation loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    # plot accuracy again epoch
    axes[1].plot(epochs, acc, 'bo', label='Training acc')
    axes[1].plot(epochs, val_acc, 'b', label='Validation acc')
    axes[1].set_title('Training and validation accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.show()

def train():
    train_generator = generator(32, 255, 255, 'training')
    image_batch_train, label_batch_train = next(train_generator)
    show_batch(image_batch_train, label_batch_train)
    valid_generator = generator(1, 255, 255, 'validation')
    # step size
    step_size_train = train_generator.n//train_generator.batch_size
    step_size_valid = valid_generator.n//valid_generator.batch_size
    model = build_cnn()
    model.summary()
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=step_size_train,
                                  validation_data=valid_generator,
                                  validation_steps=step_size_valid,
                                  epochs=10,
                                  verbose=1)
    plot_history(history)
    model.save('id_fusulinids.h5')

if __init__ == __main__:
    train()
