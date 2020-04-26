import tensorflow as tf
from keras.models import load_model
import cv2
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import numpy as np

#lis = ['A', 'B', 'Bien 60', 'D', 'Bien 30', 'Bien 70', 'Bien 20', 'Bien Bao 50', 'Bien 120', 'J']
lis = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('model.h5')

#img_test = cv2.imread('00000_00024.ppm')
#img_test = cv2.imread('00031_00014.ppm')
img_test = cv2.imread('00032_00021.ppm')
img_test = cv2.resize(img_test, (32, 32))
img_test = img_test.reshape(1, 32, 32, 3)
print(model.predict(img_test))

class_labels = np.argmax(model.predict(img_test)[0])
print('ket qua=',lis[class_labels])