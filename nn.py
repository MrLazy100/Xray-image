import os
import numpy as np
import pandas as pd 
import random
import cv2
import matplotlib.pyplot as plt
import sys
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf


inputs = Input(shape=(64, 64, 3))



x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
x = MaxPool2D(pool_size=(2, 2))(x)



x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2, 2))(x)

# Step 3 - Flattening
x = Flatten()(x)
x = Dense(units=128, activation='relu')(x)
x = Dense(units=128, activation='relu')(x)



output = Dense(units=1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

d1=sys.argv[1]
d2=sys.argv[2]
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)

training_set = train_datagen.flow_from_directory(d1,target_size = (64,64),batch_size = 32,class_mode = 'binary')

test_set = test_datagen.flow_from_directory(d2,
                                            target_size = (64,64),
                                                batch_size = 32,
                                                class_mode = 'binary')


model.fit_generator(training_set,
                             samples_per_epoch = int(len(training_set)*32),
                             nb_epoch = 1,
                             validation_data = test_set,
                             nb_val_samples = int(len(test_set)*32))
from keras.models import save_model
save_model(model,'temp.h5')