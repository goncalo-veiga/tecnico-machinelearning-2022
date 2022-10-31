import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import cv2
import shutil
from keras import optimizers
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.utils import array_to_img
from PIL import Image
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50
from keras import optimizers
from keras.optimizers import schedules
import os
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE, RandomOverSampler
from keras.regularizers import L1 as l1, L2 as l2
import tensorflow as tf
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#np.set_printoptions(threshold=sys.maxsize)
#pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')


## Load datasets
x_train = np.load('Xtrain_Classification2.npy')
y_train = np.load('Ytrain_Classification2.npy')
x_test = np.load('Xtest_Classification2.npy')

#0 - white center
#1 - rings
#2 - background

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

counter_0 = 0
counter_1 = 0
counter_2 = 0
for i in (y_train):
    if i == 0:
        counter_0 += 1
    elif i == 1:
        counter_1 += 1
    elif i == 2:
        counter_2 += 1
        
print("class 0:", counter_0, "| class 1:", counter_1, "| class 2:", counter_2)

print(np.amax(x_train))

x_train_new = []
for i in range(len(x_train)):
    x_train_new.append(x_train[i].reshape(5,5,3))
    #if y_train[i] == 0:
    #    img  = Image.fromarray(x_train_new[i])
    #    img.save("0_train/%d.png"%(i))
    #elif y_train[i] == 1:
    #    img  = Image.fromarray(x_train_new[i])
    #    img.save("1_train/%d.png"%(i))
    #elif y_train[i] == 2:
    #    img  = Image.fromarray(x_train_new[i])
    #    img.save("2_train/%d.png"%(i))

x_test_new = []
for i in range(len(x_test)):
    x_test_new.append(x_test[i].reshape(5,5,3))
    #img  = Image.fromarray(x_test_new[i])
    #img.save("x_test/%d.png"%(i))
        
x_train_new = np.array(x_train_new) 
print(x_train_new)
x_test_new = np.array(x_test_new) 
print(x_test_new)
print("sucesso")
