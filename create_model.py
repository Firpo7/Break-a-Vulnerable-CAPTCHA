#!/usr/bin/env python3

import os, keras
import numpy as np
from utils import *
from termcolor import colored
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization

print()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

name = 'new_model.h5'

(X_train, y_train, tot_train) = load_data()
(X_test, y_test, tot_test) = load_data(useTest=True)

X_train = X_train.reshape(tot_train,34,30,1)
X_test = X_test.reshape(tot_test,34,30,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(48, kernel_size=(5, 5), activation='relu', input_shape=(34,30,1)))
model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(192, activation='relu'))
model.add(Dense(26, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 128
num_classes = 26
epochs = 13

print("Start training the model: " + colored(name, "blue"))

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test),
          )

model.save(name)