import pandas as pd
from pandas import read_csv
import numpy as np
import os
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from tensorflow.keras.callbacks import ModelCheckpoint


def data():

    x_train = np.load('x.npy')
    y_train = np.load('y.npy')
    x_val = np.load('x_valid.npy')
    y_val = np.load('y_valid.npy')
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
    x_val  = x_val.reshape(x_val.shape[0], x_val.shape[1],1)
    return x_train, y_train, x_val, y_val

x_train, y_train, x_val, y_val = data()


model = Sequential()
model.add(Conv1D(80, kernel_size=3, activation ='relu', input_shape=(x_train.shape[1],1)))#16 relu
model.add(Conv1D(110, kernel_size=3, activation = 'relu', padding='same'))#2 relu
model.add(Conv1D(80, kernel_size=3, activation = 'relu', padding='same'))#2 relu
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))#256 relu
model.add(Dense(128, activation = 'relu'))#32 sig
model.add(Dense(39249, activation='linear'))
adam = keras.optimizers.Adam(learning_rate=10**-3)
print(model.summary())
model.compile(loss='mse', optimizer=adam)
filepath="model-{epoch:02d}-tloss-{loss:.3e}-vloss-{val_loss:.3e}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(x_train, y_train,
          batch_size=16,
          epochs=10000,
          verbose=2,
          validation_data=(x_val, y_val),
          callbacks=callbacks_list)
