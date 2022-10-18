
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import cv2
from keras import optimizers
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import Image
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50
from tensorflow import keras
from keras import optimizers
from keras.optimizers import schedules

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

## Load datasets
x_train = np.load('Xtrain_Classification1.npy')
y_train = np.load('ytrain_Classification1.npy')
#x_test = np.load('Xtest_Regression2.npy')

## Evaluate shape
print(y_train.shape)
print(x_train.shape)

## Dataset processing (array to image) and saves the images
x_train_new = []
count_eyespot = 0
count_spot = 0

for i in range(0, len(x_train)):
    x_train_new.append(x_train[i].reshape(30,30,3))
    #img  = Image.fromarray(x_train_new[i])
    #img.save("images/%d.png"%(i))
    if y_train[i] == 1:
        #print("EYESPOTS indices:", i)
        #img  = Image.fromarray(x_train_new[i])
        #img.save("eyespots/%d.png"%(i))
        count_eyespot += 1
    else:
        #print("SPOTS indice:", i)
        #img  = Image.fromarray(x_train_new[i])
        #img.save("spots/%d.png"%(i))
        count_spot += 1
        
print("The Images are saved successfully")         
print("Nº of Eyespots", count_eyespot)
print("Nº of Spots:", count_spot)
    
x_train_new = np.array(x_train_new)

## Solving Imbalanced Dataset (eyespots: 3131 | spot: 5142)

# - 1º Approach: Undesampling - eyespots: 3131 | spot: 3131


# - 2º Approach: Oversampling - eyespots: 5142 (3131 + 2011) | spot: 5142


# - 3º Approach: Data Augmentation - eyespots: ~~ 6500 | spot: ~~ 6500     

     
## Normalization
x_train_new = x_train_new/255


## Model CNN 

model = models.Sequential()
# 1º layer convolutional
model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(30, 30, 3)))
# add a pooling layer
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# 2º layer convolutional
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
# add another pooling layer
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Flattening layer
model.add(layers.Flatten())

# add a layer with 500 neurons
model.add(layers.Dense(500, activation='relu'))
# drop out layer
model.add(layers.Dropout(0.5))
# add another layer with 250 neurons
model.add(layers.Dense(250, activation='relu'))
# drop out layer
model.add(layers.Dropout(0.5))
# add another layer with 125 neurons
model.add(layers.Dense(125, activation='relu'))

model.add(layers.Dropout(0.6))

# add another layer with 75 neurons
model.add(layers.Dense(75, activation='relu'))

# add the last layer with 1 neurons
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['acc', f1_score])
hist = model.fit(x_train_new, y_train, batch_size=376, epochs=10, validation_split=0.2)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
## Note: EArly STOPING PREVENT OVERFITTING, callback
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()  

## convert y into one hot enconding
