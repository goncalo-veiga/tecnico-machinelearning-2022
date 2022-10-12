from pyexpat import model
from sklearn.preprocessing import normalize
from time import time
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LassoLars
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import balanced_accuracy_score as BACC
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor


## Scores between the y real and y predicted
def scores(y_real,y_pred,mode):
    
    if mode == 'r':
        mse = MSE(y_real,y_pred)
        print('The Mean Square Error is', mse)
        return mse
    
    elif mode == 'c':
        bacc = BACC(y_real,y_pred)
        print('The Balanced Accuracy is', bacc)
        return bacc
    
    elif mode == 's':
        sse = np.sum((y_real - y_pred)**2)
        print('The Sum of Squares Error is', sse)
        return sse

            
## Validation Scores inside the validation set             
def model_val_scores(model, x_train_val, x_test_val, y_train_val, y_test_val, mode):
    
    model.fit(x_train_val, y_train_val)
    y_pred = model.predict(x_test_val)
    
    if mode == 'm':
        mse = MSE(y_test_val,y_pred)
        return mse
        
    elif mode == 's':
        sse = np.sum((y_test_val - y_pred)**2)
        return sse
    
    elif mode == 'sc':
        return model.score(x_test_val, y_test_val)
    
    elif mode == 'r2':
        return r2_score(y_test_val, y_pred)
    
    else:
        print('You must define the mode input.')
    
## Load datasets
x_train = np.load('Xtrain_Regression2.npy')
y_train = np.load('Ytrain_Regression2.npy')
x_test = np.load('Xtest_Regression2.npy')

"""Analysing the Data"""
raw_data = pd.read_csv('ex2_dataset.csv')
x_features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
y = ['y']

## Data distribution plot
"""for i in x_features:
    x_values = raw_data[i].values
    mean = raw_data[i].mean()
    sns.displot(x_values, color = 'red', kind = 'kde', legend = i)
    plt.axvline(mean,0,1, color = 'black')
    plt.xlabel(i)
    plt.show()

## Data scatter plot
for j in x_features:
    ax = sns.scatterplot(x = j, y = 'y', data = raw_data, color = 'green')
    plt.show()

## Data pair plot    
pairplot = sns.pairplot(raw_data, plot_kws={'color':'green'})
plt.show()"""



""" Outlier Algorithms and Validation """

## Local Outlier Factor Algorithm
print("------ Local Outlier Factor ------")

x_train = normalize(x_train)
y_train = normalize(y_train)

model_LOF = LocalOutlierFactor(n_neighbors=50, contamination=0.2)
model_LOF_outliers = model_LOF.fit_predict(x_train)
x_train_LOF = []
y_train_LOF = []
for i in range (0, len(model_LOF_outliers)):
    if model_LOF_outliers[i] == -1:
        #print("Outlier Line:", i, x_train[i])
        print("Outlier line:", i)
    else:
        x_train_LOF.append(x_train[i])
        y_train_LOF.append(y_train[i])

x_train_LOF = np.array(x_train_LOF)
y_train_LOF = np.array(y_train_LOF)

## Isolation Forest Algorithm
print("------ ISOLATION FOREST ------")

model_IF = IsolationForest(contamination = 0.2,random_state=0)
model_IF_outliers = model_IF.fit_predict(x_train)
x_train_IF = []
y_train_IF = []
for i in range (0, len(model_IF_outliers)):
    if model_IF_outliers[i] == -1:
        #print("Outlier Line:", i, x_train[i])
        print("Outlier line:", i)
    else:
        x_train_IF.append(x_train[i])
        y_train_IF.append(y_train[i])

x_train_IF = np.array(x_train_IF)
y_train_IF = np.array(y_train_IF)

## Eliptical Envellope Algorithm
print("------ ELIPTICAL ENVELLOPE ------")

model_EE = EllipticEnvelope(contamination=0.2,random_state=0)
model_EE_outliers = model_EE.fit_predict(x_train)
x_train_EE = []
y_train_EE = []
for i in range (0, len(model_EE_outliers)):
    if model_EE_outliers[i] == -1:
        #print("Outlier Line:", i, x_train[i])
        print("Outlier line:", i)
    else:
        x_train_EE.append(x_train[i])
        y_train_EE.append(y_train[i])

x_train_EE = np.array(x_train_EE)
y_train_EE = np.array(y_train_EE)


def kfold_func(x_train,y_train):
    ## KFold validation method for any type of regression 
    n_splits = 10
    kf = KFold(n_splits, shuffle = True)
    Ridge_val_scores = []
    LinearR_val_scores = []
    Lasso_val_scores = []
    ridge_cv = RidgeCV(alphas = [2,2.1,2.2,2.23,2.25,2.27,2.3,2.4,2.5,2.6,3,3.2,3.3,3.4,3.5,3.6,4,5,6]).fit(x_train,y_train)
    print("alpha Ridge",ridge_cv.alpha_)
    lasso_cv = LassoCV(alphas = [1100,1200,1300,1400,1500,2000,2200,2300,2500,3000,5000,7000,10000,15000,1000000]).fit(x_train,y_train)
    print("alpha lasso",lasso_cv.alpha_)
    
    for train_index, test_index in kf.split(x_train):
        x_train_val, x_test_val = x_train[train_index], x_train[test_index]
        y_train_val, y_test_val = y_train[train_index], y_train[test_index]
        Ridge_val_scores.append(model_val_scores(Ridge(alpha = ridge_cv.alpha_), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        LinearR_val_scores.append(model_val_scores(LinearRegression(), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        Lasso_val_scores.append(model_val_scores(Lasso(alpha=lasso_cv.alpha_), x_train_val, x_test_val, y_train_val, y_test_val, 's'))

    print("Score: SSE | K Folds Validation nº",n_splits,"splits | Ridge:", np.average(Ridge_val_scores),"| Linear:", np.average(LinearR_val_scores),"| Lasso:", np.average(Lasso_val_scores))
    scores_list = [np.average(Ridge_val_scores), np.average(LinearR_val_scores), np.average(Lasso_val_scores)]
    minSSE = min(scores_list)
    if minSSE == scores_list[0]:
        print("-> Best is Ridge Regression:", minSSE)
    elif minSSE == scores_list[1]:
        print("-> Best is Linear Regression:", minSSE)
    else:
        print("-> Best is Lasso Regression:", minSSE)

def kfold_rob_func(x_train,y_train):
    ## KFold validation method for any type of regression 
    kf = KFold(10, shuffle = False)
    Huber_val_scores = []
    Ransac_val_scores = []
    Theilsen_val_scores = []
    
    for train_index, test_index in kf.split(x_train):
        x_train_val, x_test_val = x_train[train_index], x_train[test_index]
        y_train_val, y_test_val = y_train[train_index], y_train[test_index]
        Huber_val_scores.append(model_val_scores(HuberRegressor(alpha= 10000,epsilon=1.75), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        #Ransac_val_scores.append(model_val_scores(RANSACRegressor(), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        Theilsen_val_scores.append(model_val_scores(TheilSenRegressor(), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
    
    min = 9999
    ideal_alpha = 99999
    """
    for my_alpha in range(1,1300):
        
        for train_index, test_index in kf.split(x_train):
            x_train_val, x_test_val = x_train[train_index], x_train[test_index]
            y_train_val, y_test_val = y_train[train_index], y_train[test_index]
            Huber_val_scores.append(model_val_scores(HuberRegressor(alpha= my_alpha,epsilon=1.75), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        if np.average(Huber_val_scores) < min:
            min = np.average(Huber_val_scores)
            ideal_alpha = my_alpha
    """
    print("idela alpha:",ideal_alpha,"SSE:")
    print("Score: SSE | K Folds Validation nº",10,"splits | Huber:", np.average(Huber_val_scores),"| Ransac:", np.average(Ransac_val_scores),"| Theilsen:", np.average(Theilsen_val_scores))



print("----LOF SSE----")
kfold_func(x_train_LOF,y_train_LOF.ravel())
print("----IF SSE----")
kfold_func(x_train_IF,y_train_IF.ravel())
print("----EE SSE----")
kfold_func(x_train_EE,y_train_EE.ravel())




"""
model_huber = HuberRegressor(epsilon=1.5)
model_huber_fit = model_huber.fit(x_train, y_train.ravel())
y_train_pred = model_huber.predict(x_train)
y_test = model_huber.predict(x_test)
print(model_huber.outliers_)
count = 0
for i in range(0,len(model_huber.outliers_)):
    if model_huber.outliers_[i]:
        count +=1
print("counter:",count)

print("----Huber SSE----")
kfold_rob_func(normalize(x_train),normalize(y_train))

"""