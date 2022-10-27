
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import cv2
import os
from keras import optimizers
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.regularizers import L1 as l1, L2 as l2
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50
from keras import optimizers
from keras.optimizers import schedules
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#np.set_printoptions(threshold=sys.maxsize)
#pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')


## This function returns the recall for F1 Score
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

## This function returns the precision for F1 Score
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

## This function returns the F1 Score for the classification
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

## Training dataset to (30,30,3) shape
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
        
## Test dataset to (30,30,3) shape
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
# - 1º Approach: Oversampling - eyespots: 5142 (3131 + 2011) | spot: 5142
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

x_oversampling_new = np.array(x_oversampling_new)
print(x_oversampling_new.shape)
x_oversampling_new, y_oversampling = shuffle(x_oversampling_new, y_oversampling)

# - 3º Approach: Data Augmentation - eyespots: ~~ same | spot: ~~ same


# - 4º Approach: Class weight adjustement
    # Eyepots -> 1 -> 3131 -> Class weight : 1.66
    # Spots -> 0 -> 5142 -> Class weight : 1
    # Results: Not worth

## Data Normalization 
x_train_new = x_train_new/255
x_oversampling_new = x_oversampling_new/255
x_test = x_test/255

## Split into Training and Validation from scratch
train_X,valid_X,train_label,valid_label = train_test_split(x_oversampling_new, y_oversampling, test_size=0.165, random_state=50)
#train_X,valid_X,train_label,valid_label = train_test_split(x_train_new, y_train, test_size=0.165, random_state=14)
print("Training shape x  e  y:",train_X.shape, train_label.shape) 
print("Validation shape x e y :",valid_X.shape, valid_label.shape)


##  Model: Convolutional Neural Network
#callback = keras.callbacks.EarlyStopping(monitor='val_acc', patience=200, verbose=1)
model = models.Sequential([
    layers.Conv2D(32, (5,5), padding = 'same', activation = 'relu', input_shape=(30,30,3)),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Conv2D(16, (5,5), padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation = "relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation = 'sigmoid'),
])

## Compile the model 
model.compile(optimizer = 'adam' , loss = 'binary_crossentropy',metrics = ['acc'])

## Fit the training data into the model
hist = model.fit(train_X, train_label, batch_size=64, epochs=25)
#hist = model.fit(x_oversampling_new, y_oversampling, batch_size=64, epochs=25, validation_split=0.165)

## Predict for validation data
valid_pred = model.predict(valid_X)

## Evaluating the classification
target_names = ['spots 0', 'eyespots 1']
print(classification_report(valid_label, np.round(abs(valid_pred)), target_names = target_names))
print(confusion_matrix(valid_label, np.round(abs(valid_pred))))

## Plots with loss and accuracy (training and validatino)
"""
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

## Predictions
#predictions = model.predict(x_test_new)
#predictions_tc = to_categorical(predictions, num_classes = 2)
#for i in range(0,50):
#    print("predictions pura", predictions[i], i)
    #print("predictions", predictions_tc[i], i+1)


#plot_confusion_matrix(model, test_data, test_labels)

## Training and Validation Accuracy
#plt.plot(epochs, f1_acc, 'bo', label='F1 score Training acc')
#plt.plot(epochs, val_f1_acc, 'r', label='F1 score Validation acc')
#plt.title('F1 score Training and Validation acc')
#plt.legend()

#plt.show()

## convert y into one hot enconding"""
