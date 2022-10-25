
from audioop import avg
import sys
from tabnanny import verbose
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
import os
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE, RandomOverSampler
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

np.set_printoptions(threshold=sys.maxsize)
#pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')

#config = tf.ConfigProto(
##        device_count = {'GPU': 0}
#    )
#sess = tf.Session(config=config)


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
x_test = np.load('Xtest_Classification1.npy')

## Evaluate shape
print(y_train.shape)
print(x_train.shape)
print(x_test.shape)

## Dataset processing (array to image) and saves the images
x_train_new = []
x_test_new = []
count_eyespot_train = 0
count_spot_train = 0

## TRAINING SET
for i in range(0, len(x_train)):
    x_train_new.append(x_train[i].reshape(30,30,3))
    #img  = Image.fromarray(x_train_new[i])
    #img.save("images/%d.png"%(i))
    if y_train[i] == 1:
        #print("EYESPOTS indices:", i)
        #img  = Image.fromarray(x_train_new[i])
        #img.save("eyespots/%d.png"%(i))
        count_eyespot_train += 1
    else:
        #print("SPOTS indice:", i)
        #img  = Image.fromarray(x_train_new[i])
        #img.save("spots/%d.png"%(i))
        count_spot_train += 1
        
## TEST SET
for i in range(0, len(x_test)):
    x_test_new.append(x_test[i].reshape(30,30,3))
    #img  = Image.fromarray(x_test_new[i])
    #img.save("testset/%d.png"%(i))
        
print("The Images are saved successfully")         
print("TRAIN SET | Nº of Eyespots", count_eyespot_train)
print("TRAIN SET |Nº of Spots:", count_spot_train)       

x_train_new = np.array(x_train_new)
print(x_train_new.shape)

x_test_new = np.array(x_test_new)
print(x_test_new.shape)
## Solving Imbalanced Dataset (eyespots: 3131 | spot: 5142)

# - 1º Approach: Undesampling - eyespots: 3131 | spot: 3131


# - 2º Approach: Oversampling - eyespots: 5142 (3131 + 2011) | spot: 5142

df_y_train = pd.DataFrame({'y_train': y_train.tolist()})
df_x_train = pd.DataFrame({'x_train': x_train_new.tolist()})
print(df_y_train.value_counts())

smote = RandomOverSampler(sampling_strategy='minority')
x_oversampling, y_oversampling = smote.fit_resample(x_train, df_y_train)

print("Nº DE SPOTS E EYESPOTS:", y_oversampling.value_counts())
y_oversampling = np.array(y_oversampling)

x_oversampling_new = []
for i in range(0, len(x_oversampling)):
    x_oversampling_new.append(x_oversampling[i].reshape(30,30,3))
    if y_oversampling[i] == 1:
        #print("EYESPOTS indices:", i)
        img  = Image.fromarray(x_oversampling_new[i])
        img.save("oversampling/eyespots/%d.png"%(i))
        count_eyespot_train += 1
    else:
        #print("SPOTS indice:", i)
        img  = Image.fromarray(x_oversampling_new[i])
        img.save("oversampling/spots/%d.png"%(i))
        count_spot_train += 1
 
#x_oversampling_new, y_oversampling = shuffle(x_oversampling_new, y_oversampling, random_state = 0) 
print("gravacao done")  
print("Y OVER", y_oversampling)
x_oversampling_new = np.array(x_oversampling_new)
print(x_oversampling_new.shape)

x_oversampling_new, y_oversampling = shuffle(x_oversampling_new, y_oversampling)

# - 3º Approach: Data Augmentation - eyespots: ~~ 6500 | spot: ~~ 6500     

## Normalization and shuffling
x_train_new = x_train_new/255

x_oversampling_new = x_oversampling_new/255

##  Convolutional Neural Network Model
callback = keras.callbacks.EarlyStopping(monitor='val_acc', patience=8, verbose=1)
# This callback will stop the training when there is no improvement in
# the loss for 2 consecutive epochs.

print("VETOR DO Y", y_oversampling)

model = models.Sequential()

## FIRST CNN
"""# 1º layer convolutional
model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(30, 30, 3)))
# add a pooling layer
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# 2º layer convolutional
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
# add another pooling layer
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Flattening layer
model.add(layers.Flatten()) 

# add a layer with 500 neurons))
model.add(layers.Dense(500, activation='relu'))
# drop out layer
model.add(layers.Dropout(0.5))
# add another layer with 250 neurons
model.add(layers.Dense(250, activation='relu'))
# drop out layer
model.add(layers.Dropout(0.5))
# add another layer with 125 neurons
model.add(layers.Dense(125, activation='relu'))
# drop out layer
model.add(layers.Dropout(0.6))
# add another layer with 75 neurons
model.add(layers.Dense(75, activation='relu'))
# add the last layer with 1 neurons
model.add(layers.Dense(1, activation='sigmoid'))"""
## ----------------------------------------------------------------

## SECOND CNN
model.add(layers.Conv2D(32, (5,5), padding = 'same', activation = 'relu', input_shape=(30,30,3)))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (5,5), padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(1, activation = 'sigmoid'))


## Third CNN
"""model.add(layers.Conv2D(64, (6,6), padding = 'same', activation = 'relu', input_shape=(30,30,3)))
model.add(layers.Conv2D(64, (6,6), padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D(padding = 'same'))
model.add(layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'))
model.add(layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D(padding = 'same'))
#model.add(layers.Conv2D(128, (5,5), padding = 'same', activation = 'relu'))
#model.add(layers.MaxPooling2D(padding = 'same'))
model.add(layers.Flatten())
model.add(layers.Dense(1254, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))"""

model.summary()


model.compile(optimizer = optimizers.Adam() , loss = 'binary_crossentropy',metrics = ['acc'])
hist = model.fit(x_oversampling_new, y_oversampling, batch_size=64, epochs=25, validation_split=0.16, callbacks = [callback])

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
#f1_acc = hist.history['f1_score']
#val_f1_acc = hist.history['val_f1_score']

epochs = range(1, len(acc) + 1)
## Traning and Validation Accuracy
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

## Training and Validation Loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()  

## Training and Validation Accuracy
#plt.plot(epochs, f1_acc, 'bo', label='F1 score Training acc')
#plt.plot(epochs, val_f1_acc, 'r', label='F1 score Validation acc')
#plt.title('F1 score Training and Validation acc')
#plt.legend()

#plt.show()

## convert y into one hot enconding 
