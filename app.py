from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np
import keras
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Conv2D, Reshape
from keras.layers import MaxPooling2D, Activation, Dropout
from keras import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st
from matplotlib import pyplot as plt

model = tf.keras.models.load_model('./model/id_fusulinids.h5')
