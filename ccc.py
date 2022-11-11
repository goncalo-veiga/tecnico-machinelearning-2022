"""
Authors:
Gonçalo Veiga - 96738
Gonçalo Galante - 96216
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
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
    
    
x_train_patches = []

train_x,valid_x,train_label,valid_label = train_test_split(x_train, y_train, test_size=0.2, random_state=9)
"""

##  1) Neural Network - Multilayer Perceptron
#callback = keras.callbacks.EarlyStopping(monitor='val_acc', patience=200, verbose=1)
model = models.Sequential([
    layers.Flatten(input_shape=(5,5,3)),
    layers.Dense(100, activation = "relu"),
    layers.Dense(100, activation = 'relu'),
    layers.Dense(50, activation = 'relu'),
    layers.Dense(25, activation = 'relu'),
    layers.Dense(3, activation = 'softmax'),
])
model.summary()

## Compile the model 
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy',metrics = ['acc'])

## Fit the training data into the model
#hist = model.fit(train_x, train_label, batch_size=64, epochs=25)
hist = model.fit(x_oversampling_new, y_oversampling, batch_size=64, epochs=25, validation_data=(valid_x_oversampling, valid_label))

valid_pred = model.predict(valid_x_oversampling)

print("label original", np.argmax(valid_label, axis=-1))
print("label predicted", np.argmax(valid_pred, axis=-1))

print("Validation Balanced Accuracy Score:", balanced_accuracy_score(np.argmax(valid_label, axis=-1), np.argmax(valid_pred, axis=-1)))
print("Validation Accuracy: ", accuracy_score(np.argmax(valid_label, axis=-1), np.argmax(valid_pred, axis=-1)))"""

"""target_names = ['class 0', 'class 1', 'class 2'] 
#print(classification_report(np.argmax(valid_label, axis=-1),np.argmax(valid_pred, axis=-1), target_names = target_names))
#print(confusion_matrix(np.argmax(valid_label, axis=-1),np.argmax(valid_pred, axis=-1)))
## Plots with loss and accuracy (training and validatino)
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

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

plt.show() """


df_y_train = pd.DataFrame({'y_train': train_label.tolist()})
df_x_train = pd.DataFrame({'x_train': train_x.tolist()})

smote = RandomOverSampler()
x_oversampling, y_oversampling = smote.fit_resample(train_x, df_y_train)

print("Data Oversampling:", y_oversampling.value_counts())
y_oversampling = np.array(y_oversampling)
x_oversampling, y_oversampling = shuffle(x_oversampling, y_oversampling, random_state=9)
# Oversampling full data
df_y_train_final = pd.DataFrame({'y_train': y_train.tolist()})
df_x_train_final = pd.DataFrame({'x_train': x_train.tolist()})

smote = RandomOverSampler()
x_oversampling_final, y_oversampling_final = smote.fit_resample(x_train, df_y_train_final)

print("Full Data Oversampling:", y_oversampling_final.value_counts())
y_oversampling_final = np.array(y_oversampling_final)
print("Shape: ",x_oversampling_final.shape)
x_oversampling_final, y_oversampling_final = shuffle(x_oversampling_final, y_oversampling_final, random_state=9)



## data normalization
train_x = train_x/255
valid_x = valid_x/255
x_oversampling = x_oversampling/255
x_oversampling_final = x_oversampling_final/255
x_test = x_test/255
#x_oversampling_new = x_oversampling_new/255

## 2) Support Vector Machine
def method_svc(x_data,y_data):
    #svc = SVC(decision_function_shape='ovo', verbose = True, random_state=9, class_weight='balanced') # class weights
    svc = SVC(decision_function_shape='ovo', verbose = True, random_state=9, class_weight='balanced') # oversampling
    svc.fit(x_data, y_data)
    valid_pred = svc.predict(valid_x)
    svc_bacc = balanced_accuracy_score((valid_label), valid_pred)
    print(" SVC | Validation Balanced Accuracy Score:", svc_bacc)
    #print(" SVC | Validation Accuracy: ", accuracy_score(valid_label, valid_pred))
    return svc_bacc


## 3) Decision Tree
def method_dt(x_data,y_data):
    clf = DecisionTreeClassifier(criterion = 'entropy', random_state=9, class_weight='balanced')
    clf.fit(x_data, y_data)
    valid_pred_dt = clf.predict(valid_x)
    dt_bacc = balanced_accuracy_score((valid_label), valid_pred_dt)
    print(" DT | Validation Balanced Accuracy Score:", dt_bacc)
    #print(" DT |Validation Accuracy: ", accuracy_score(valid_label, valid_pred_dt))
    return dt_bacc

## 4) Linear Support Vector Machine 
def method_lsvc(x_data,y_data):
    lsvc = LinearSVC(class_weight='balanced')
    lsvc.fit(x_data, y_data)
    valid_pred_lsvc = lsvc.predict(valid_x)
    lsvc_bacc = balanced_accuracy_score((valid_label), valid_pred_lsvc)
    print("LSVC | Validation Balanced Accuracy Score:", lsvc_bacc)
    #print("LSVC | Validation Accuracy: ", accuracy_score(valid_label, valid_pred_lsvc))
    return lsvc_bacc

## 5) k-Nearest Neighbors
def method_knn(x_data,y_data,numb_neigh):
    # 4 n-neigh is best
    neigh = KNeighborsClassifier(n_neighbors=numb_neigh, weights='uniform')
    neigh.fit(x_data, y_data)
    valid_pred_knn = neigh.predict(valid_x)
    knn_bacc = balanced_accuracy_score((valid_label), valid_pred_knn)
    print(" KNN | Validation Balanced Accuracy Score:", knn_bacc)
    #print(" KNN | Validation Accuracy: ", accuracy_score(valid_label, valid_pred_knn))
    return knn_bacc

## 6) Random Forest
def method_rf(x_data,y_data):
    rf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    rf.fit(x_data, y_data)
    valid_pred_rf = rf.predict(valid_x)
    rf_bacc = balanced_accuracy_score((valid_label), valid_pred_rf)
    print(" RF | Validation Balanced Accuracy Score:", rf_bacc)
    #print(" RF | Validation Accuracy: ", accuracy_score(valid_label, valid_pred_knn))
    return rf_bacc

## 6) Random Forest
def method_ada(x_data,y_data):
    ada = AdaBoostClassifier()
    ada.fit(x_data, y_data)
    valid_pred_ada = ada.predict(valid_x)
    ada_bacc = balanced_accuracy_score((valid_label), valid_pred_ada)
    print(" ADA | Validation Balanced Accuracy Score:", ada_bacc)
    #print(" ADA | Validation Accuracy: ", accuracy_score(valid_label, valid_pred_knn))
    return ada_bacc

## 7) Gaussian Naive Bayes
def method_gnb(x_data,y_data):
    gnb = GaussianNB()
    gnb.fit(x_data, y_data)
    valid_pred_gnb = gnb.predict(valid_x)
    gnb_bacc = balanced_accuracy_score((valid_label), valid_pred_gnb)
    print(" GNB | Validation Balanced Accuracy Score:", gnb_bacc)
    #print(" GNB | Validation Accuracy: ", accuracy_score(valid_label, valid_pred_knn))
    return gnb_bacc










#method_svc(train_x,train_label)
#method_lsvc(train_x,train_label)
#method_dt(train_x,train_label)
#method_knn(train_x,train_label,4)
method_rf(train_x,train_label)

method_ada(train_x,train_label)
method_gnb(train_x,train_label)


