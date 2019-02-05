import os
import cv2
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from model import *
from data import *

batch_size = 8
epochs = 10

model = fcn()

train_gen = data_generator('./dataset/train', batch_size)
validation_data = val_data('./dataset/validation')

train_len = len(glob.glob(os.path.join('./dataset/train', 'data/*.jpg')))

LOG_FILE_PATH = './checkpoint/checkpoint-{epoch:02d}-{val_acc:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=LOG_FILE_PATH, monitor='val_acc', verbose=1, save_best_only=True)

history = model.fit_generator(train_gen, steps_per_epoch=train_len/batch_size,
epochs=epochs, verbose=1, validation_data=validation_data, callbacks=[checkpoint])

model.trainable = False
model.save('model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
