
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


## Function to define F1 Score
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def f1_score_(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

## Load datasets
x_train = np.load('Xtrain_Classification1.npy')
y_train = np.load('ytrain_Classification1.npy')
x_test = np.load('Xtest_Classification1.npy')

## Reshape Train Dataset to (n,30,30,3) and anaylising the output occurences
x_train_new = []
count_eyespot_train = 0
count_spot_train = 0
eyespots_index = []
spots_index = []
for i in range(len(x_train)):
    x_train_new.append(x_train[i].reshape(30,30,3))
    if y_train[i] == 1:
        eyespots_index.append(i)
        #img  = Image.fromarray(x_train_new[i])
        #img.save("eyespots/%d.png"%(i))
        count_eyespot_train += 1
    else:
        spots_index.append(i)
        #img  = Image.fromarray(x_train_new[i])
        #img.save("spots/%d.png"%(i))
        count_spot_train += 1
x_train_new = np.array(x_train_new) # X_Train array reshaped (n,30,30,3)

## Reshape Test Dataset to (n,30,30,3)
x_test_new = []
for i in range(len(x_test)):
    x_test_new.append(x_test[i].reshape(30,30,3))
    
x_test_new = np.array(x_test_new) # X_Test array reshaped (n,30,30,3)

""" Validation Phase: Splits and solving Dataset Imbalanced """

## Split into Training and Validation from scratch for Imbalanced Dataset
#train_x,valid_x,train_label,valid_label = train_test_split(x_train_new, y_train, test_size=0.165, random_state = 29)
#print("TRAINING shape x  e  y:",train_x.shape, train_label.shape)
#print("VALIDATION shape x e y :",valid_x.shape, valid_label.shape)
#train_x, train_label = shuffle(train_x, train_label, random_state=29)

## Split into Training and Validation from scratch for Oversampling
train_x,valid_x,train_label,valid_label = train_test_split(x_train, y_train, test_size=0.165, random_state=29)

valid_x_oversampling = []
for i in range(len(valid_x)):
    valid_x_oversampling.append(valid_x[i].reshape(30,30,3))

valid_x_oversampling = np.array(valid_x_oversampling)
#print("TRAINING shape x  e  y:",train_x.shape, train_label.shape)
#print("VALIDATION shape x e y :",valid_x.shape, valid_label.shape)


## Solving Imbalanced Dataset (eyespots: 3131 | spot: 5142)

# - 1º Approach: Oversampling - eyespots: 5142 (3131 + 2011) | spot: 5142
    ## RESULTS: not good results overall
"""df_y_train = pd.DataFrame({'y_train': train_label.tolist()})
df_x_train = pd.DataFrame({'x_train': train_x.tolist()})
print("Normal Dataset:", df_y_train.value_counts())

smote = RandomOverSampler(sampling_strategy='minority')
x_oversampling, y_oversampling = smote.fit_resample(train_x, df_y_train)

print("Data Oversampling:", y_oversampling.value_counts()) #5142 e 5132
y_oversampling = np.array(y_oversampling)

x_oversampling_new = []
for i in range(len(x_oversampling)):
    x_oversampling_new.append(x_oversampling[i].reshape(30,30,3))
    if y_oversampling[i] == 1:
        #print("EYESPOTS indices:", i)
        #img  = Image.fromarray(x_oversampling_new[i])
        #img.save("oversampling/eyespots/%d.png"%(i))
        count_eyespot_train += 1
    else:
        #print("SPOTS indice:", i)
        #img  = Image.fromarray(x_oversampling_new[i])
        #img.save("oversampling/spots/%d.png"%(i))
        count_spot_train += 1

x_oversampling_new = np.array(x_oversampling_new)
#print(x_oversampling_new.shape)
x_oversampling_new, y_oversampling = shuffle(x_oversampling_new, y_oversampling, random_state = 13)"""

# - 2º Approach: Data Augmentation
    ## RESULTS: better results overall
"""#datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,fill_mode="nearest")
datagen = ImageDataGenerator(zoom_range=0.1, fill_mode="nearest")
#eyespots 1
for n in eyespots_index[0:1000]:
    img = load_img('trainingset/1eyespots/%d.png'%(n))
    np_x = img_to_array(img)
    #print("first shape",np_x.shape)
    np_x = np_x.reshape((1,) + np_x.shape)
    #print("second shape", np_x.shape)
    i = 0
    for batch in datagen.flow(np_x, batch_size=1, save_to_dir='preview',save_prefix='augmented',save_format='png'):
        i += 1
        if i > 5:
            break
ctr = 0
for images in os.listdir("preview/"):
    if (images.endswith(".png")):
        img = load_img("preview/" + images)
        #print(images)
        np_x = img_to_array(img)
        train_x = np.append(train_x,[np_x],axis = 0)
        ctr += 1
        if ctr == 1671:
            break
        
y_aug = np.ones([1671,])
train_label = np.append(train_label,y_aug)

counter_eyespots = 0
counter_spots = 0
for i in range(0,len(train_label)):
    if train_label[i]==1:
        counter_eyespots+=1
    else:
        counter_spots+=1

print("NUMERO DE EYESPOTS  E SPOTS NO TREINO",counter_eyespots, counter_spots)       

print("SHAPE DO TREINO", train_x.shape)
print("SHAPE DO TREINO Y", train_label.shape)

train_x, train_label = shuffle(train_x, train_label, random_state=13)"""

# - 3º Approach: Class Weights
    ## RESULTS: worse results 
  # class 0, weight 1
  # class 1, weight 1.66

# - 4º Approach: Oversampling + Data Augmentation
    # RESULTS: best results
"""df_y_train = pd.DataFrame({'y_train': train_label.tolist()})
df_x_train = pd.DataFrame({'x_train': train_x.tolist()})
print("Normal Dataset:", df_y_train.value_counts())

smote = RandomOverSampler(sampling_strategy='minority')
x_oversampling, y_oversampling = smote.fit_resample(train_x, df_y_train)

print("Data Oversampling:", y_oversampling.value_counts()) #4289 4289
y_oversampling = np.array(y_oversampling)

x_oversampling_new = []
for i in range(len(x_oversampling)):
    x_oversampling_new.append(x_oversampling[i].reshape(30,30,3))
    if y_oversampling[i] == 1:
        #print("EYESPOTS indices:", i)
        #img  = Image.fromarray(x_oversampling_new[i])
        #img.save("oversampling/eyespots/%d.png"%(i))
        count_eyespot_train += 1
    else:
        #print("SPOTS indice:", i)
        #img  = Image.fromarray(x_oversampling_new[i])
        #img.save("oversampling/spots/%d.png"%(i))
        count_spot_train += 1

x_oversampling_new = np.array(x_oversampling_new)
#print(x_oversampling_new.shape)
x_oversampling_new, y_oversampling = shuffle(x_oversampling_new, y_oversampling, random_state = 29)

counter_eyespots = 0
counter_spots = 0
for i in range(0,len(y_oversampling)):
    if y_oversampling[i]==1:
        counter_eyespots+=1
    else:
        counter_spots+=1

print("Nº eyespots e spots oversampling:",counter_eyespots, counter_spots)     

datagen = ImageDataGenerator(zoom_range=0.1, fill_mode="nearest")

try:
    shutil.rmtree("C:/Users/gonca/Desktop/class/aug0")
except OSError:
    print("No directory to delete")
    
try:
    shutil.rmtree("C:/Users/gonca/Desktop/class/aug1")
except OSError:
    print("No directory to delete")

os.makedirs("C:/Users/gonca/Desktop/class/aug0")
os.makedirs("C:/Users/gonca/Desktop/class/aug1")

for n in eyespots_index[0:1000]:
    img = load_img('trainingset/1eyespots/%d.png'%(n))
    np_x = img_to_array(img)
    #print("first shape",np_x.shape)
    np_x = np_x.reshape((1,) + np_x.shape)
    #print("second shape", np_x.shape)
    i = 0
    for batch in datagen.flow(np_x, batch_size=1, save_to_dir='aug1',save_prefix='aug_eyespot',save_format='png'):
        i += 1
        if i > 5:
            break

ctr = 0
for images in os.listdir("aug1/"):
    if (images.endswith(".png")):
        img = load_img("aug1/" + images)
        #print(images)
        np_x = img_to_array(img)
        x_oversampling_new = np.append(x_oversampling_new,[np_x],axis = 0)
        ctr += 1
        if ctr == 2500:
            break        

y_aug1 = np.ones([2500,])
y_oversampling = np.append(y_oversampling,y_aug1)

for n in spots_index[0:1000]:
    img = load_img('trainingset/0spots/%d.png'%(n))
    np_x = img_to_array(img)
    #print("first shape",np_x.shape)
    np_x = np_x.reshape((1,) + np_x.shape)
    #print("second shape", np_x.shape)
    i = 0
    for batch in datagen.flow(np_x, batch_size=1, save_to_dir='aug0',save_prefix='aug_spot',save_format='png'):
        i += 1
        if i > 5:
            break

ctr = 0
for images in os.listdir("aug0/"):
    if (images.endswith(".png")):
        img = load_img("aug0/" + images)
        #print(images)
        np_x = img_to_array(img)
        x_oversampling_new = np.append(x_oversampling_new,[np_x],axis = 0)
        ctr += 1
        if ctr == 2500:
            break

y_aug0 = np.zeros([2500,])
y_oversampling = np.append(y_oversampling,y_aug0)

x_oversampling_new, y_oversampling = shuffle(x_oversampling_new, y_oversampling, random_state=29)

print("shape x os + aug", x_oversampling_new.shape)
print("shape y os + aug", y_oversampling.shape)

counter_eyespots = 0
counter_spots = 0
for i in range(0,len(y_oversampling)):
    if y_oversampling[i]==1:
        counter_eyespots+=1
    else:
        counter_spots+=1

print("nº de eyespots e spots antes da norm", counter_eyespots, counter_spots)"""

"""------------------------- END OF VALIDATION PHASE --------------------------"""


""" Data Preparing to Test set Prediction Phase: solving Dataset Imbalanced """
## Approach chosen - Oversampling + Data Augmentation

## Oversampling
df_y_train = pd.DataFrame({'y_train': y_train.tolist()})
df_x_train = pd.DataFrame({'x_train': x_train.tolist()})
print("Normal Dataset:", df_y_train.value_counts())

smote = RandomOverSampler(sampling_strategy='minority')
x_oversampling, y_oversampling = smote.fit_resample(x_train, df_y_train)

print("Data Oversampling:", y_oversampling.value_counts())
y_oversampling = np.array(y_oversampling)

x_oversampling_new = []
for i in range(len(x_oversampling)):
    x_oversampling_new.append(x_oversampling[i].reshape(30,30,3))
    if y_oversampling[i] == 1:
        count_eyespot_train += 1
    else:
        count_spot_train += 1

x_oversampling_new = np.array(x_oversampling_new)
#print(x_oversampling_new.shape)
x_oversampling_new, y_oversampling = shuffle(x_oversampling_new, y_oversampling, random_state = 29)

counter_eyespots = 0
counter_spots = 0
for i in range(0,len(y_oversampling)):
    if y_oversampling[i]==1:
        counter_eyespots+=1
    else:
        counter_spots+=1

print("Nº eyespots and spots after Oversampling:",counter_eyespots, counter_spots)  

## Data augmentation
datagen = ImageDataGenerator(zoom_range=0.1, fill_mode="nearest")

directory = os.getcwd()
try:
    shutil.rmtree(directory + '/final_aug0')
except OSError:
    print("No directory to delete")
    
try:
    shutil.rmtree(directory + '/final_aug1')
except OSError:
    print("No directory to delete")

os.makedirs(directory + '/final_aug0')
os.makedirs(directory + '/final_aug1')

for n in eyespots_index[0:1000]:
    img = load_img('trainingset/1eyespots/%d.png'%(n))
    np_x = img_to_array(img)
    #print("first shape",np_x.shape)
    np_x = np_x.reshape((1,) + np_x.shape)
    #print("second shape", np_x.shape)
    i = 0
    for batch in datagen.flow(np_x, batch_size=1, save_to_dir='final_aug1',save_prefix='final_aug_eyespot',save_format='png'):
        i += 1
        if i > 5:
            break

ctr = 0
for images in os.listdir("final_aug1/"):
    if (images.endswith(".png")):
        img = load_img("final_aug1/" + images)
        #print(images)
        np_x = img_to_array(img)
        x_oversampling_new = np.append(x_oversampling_new,[np_x],axis = 0)
        ctr += 1
        if ctr == 2500:
            break        

y_aug1 = np.ones([2500,])
y_oversampling = np.append(y_oversampling,y_aug1)

for n in spots_index[0:1000]:
    img = load_img('trainingset/0spots/%d.png'%(n))
    np_x = img_to_array(img)
    #print("first shape",np_x.shape)
    np_x = np_x.reshape((1,) + np_x.shape)
    #print("second shape", np_x.shape)
    i = 0
    for batch in datagen.flow(np_x, batch_size=1, save_to_dir='final_aug0',save_prefix='final_aug_spot',save_format='png'):
        i += 1
        if i > 5:
            break

ctr = 0
for images in os.listdir("final_aug0/"):
    if (images.endswith(".png")):
        img = load_img("final_aug0/" + images)
        #print(images)
        np_x = img_to_array(img)
        x_oversampling_new = np.append(x_oversampling_new,[np_x],axis = 0)
        ctr += 1
        if ctr == 2500:
            break

y_aug0 = np.zeros([2500,])
y_oversampling = np.append(y_oversampling,y_aug0)

x_oversampling_new, y_oversampling = shuffle(x_oversampling_new, y_oversampling, random_state=29)

#print("shape x os + aug", x_oversampling_new.shape)
#print("shape y os + aug", y_oversampling.shape)

counter_eyespots = 0
counter_spots = 0
for i in range(0,len(y_oversampling)):
    if y_oversampling[i]==1:
        counter_eyespots+=1
    else:
        counter_spots+=1

print("Nº eyespots and spots after Data Augmentation", counter_eyespots, counter_spots)

## Data normalization for imbalanced dataset validation
x_train_new = x_train_new/255
x_test_new = x_test_new/255

# Data normalization for oversampled dataset validation
train_x = train_x/255
valid_x = valid_x/255

x_oversampling_new = x_oversampling_new/255
valid_x_oversampling = valid_x_oversampling/255

##  Convolutional Neural Network Models
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
hist = model.fit(x_oversampling_new, y_oversampling, batch_size=64, epochs=25)

## Predict for validation data
#valid_pred = model.predict(valid_x_oversampling)

## Predict for the test data
y_pred = model.predict(x_test_new)

## Evaluating the classification (Validation)
#target_names = ['spots 0', 'eyespots 1']
#print(classification_report(valid_label, np.round(abs(valid_pred)), target_names = target_names))
#print(confusion_matrix(valid_label, np.round(abs(valid_pred))))

#print("f1 score:", f1_score(valid_label, np.round(abs(valid_pred))))

y_test = (np.rint(abs(y_pred))).astype(int)
np.save('y_test1', y_test)

print("y pred", np.round(abs(y_pred)))
print("y pred", y_test)


"""## Plots with loss and accuracy (training and validatino)
acc = hist.history['acc']
#val_acc = hist.history['val_acc']
loss = hist.history['loss']
#val_loss = hist.history['val_loss']
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

plt.show() """


