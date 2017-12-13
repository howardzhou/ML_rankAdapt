'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from env_rankOpt import SpecEnv_rankOpt
from keras.activations import sigmoid
import time
import numpy as np

# def my_sign(x):
# 	return round(sigmoid(x))

NAP = 9
NUE = 100
interdist = 100
NTRAIN = 100000
NTEST = 1000

batch_size = 128
num_classes = NAP*NUE
epochs = 2

st_env = time.time()

env = SpecEnv_rankOpt(NAP, NUE, NTRAIN, NTEST, interdist)
# the data, shuffled and split between train and test sets
x_train, y_train, x_test, y_test = env.topoInit()

et_env = time.time()
print('Environment Initialization Finished:',et_env-st_env)
x_train = x_train.reshape(NTRAIN, NAP*NUE)
x_test = x_test.reshape(NTEST, NAP*NUE)
y_train = y_train.reshape(NTRAIN, NAP*NUE)
y_test = y_test.reshape(NTEST, NAP*NUE)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(NAP*NUE,)))
model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', # mean_squared_error binary_crossentropy
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
prd_y = model.predict(x_test[0:3])
print(np.round(prd_y))
print(y_test[0:3])
print(x_test[0:3])
print('Test loss:', score[0])
print('Test accuracy:', score[1])
et_all = time.time()
print('Total time:',et_all-st_env)
