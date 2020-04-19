from django.shortcuts import render
from django.views import View
# Create your views here.

import tensorflow as tf
from keras.models import load_model
import cv2
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import numpy as np

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def handle_uploaded_file(f, file_name):
    with open(file_name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

class IndexView(View):
    def get(self,  request):
        
        return render(request,'index.html')

    def post(self, request):
        filew = request.FILES['file']
        handle_uploaded_file(filew, filew.name)
        lis = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        print('path=',BASE_DIR )
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model = load_model(BASE_DIR+'/model1.h5')

        img_test = cv2.imread(BASE_DIR+'/'+filew.name) #chu B
        #img_test = cv2.imread('img018-00003.png') #chu H
        #img_test = cv2.imread('img020-00003.png') #chu J
        img_test = cv2.resize(img_test, (64, 64))
        img_test = img_test.reshape(1, 64, 64, 3)
        # print(model.predict(img_test))

        class_labels = np.argmax(model.predict(img_test)[0])
        kq = lis[class_labels]
        return render(request,'index.html', {'ketqua': kq})
