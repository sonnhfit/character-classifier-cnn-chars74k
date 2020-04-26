#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
from keras.utils import np_utils
import numpy as np 
from imutils import paths
# import cv/2
import os
from sklearn import preprocessing
# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()


# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)


data = []
labels = []

imagePaths = list(paths.list_images('./data'))

import os

path = './data'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.ppm' in file:
            files.append(os.path.join(r, file))
print(files)
# loop over the image paths
for imagePath in files:
    # print(imagePaths)
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath, 1)
    image = cv2.resize(image, (32, 32))
    # image = image/255.0
    # fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
    #                 cells_per_block=(1, 1), visualize=True, multichannel=True)
    # print(image)
    data.append(image)
    labels.append(label)
# print(labels)





le = preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
#print(labels)

data = np.array(data)
x_train = data.astype('float32')

x_train /= 255
y_train = np_utils.to_categorical(labels, 10)
print(y_train)
img_width, img_height = 32, 32

nb_train_samples = 5000
nb_validation_samples = 150
epochs = 150
batch_size = 16

# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_width, img_height)
# else:
input_shape = (img_width, img_height, 3)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(300, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(optimizer='sgd', loss='mean_squared_error',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs)
model.save('model.h5')

