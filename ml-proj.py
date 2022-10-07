"""
Authors: 
96738 - Gonçalo Veiga 
96216 - Gonçalo Galante
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score, KFold

## This function returns MSE, SSE and R^2 scores
def scores(y_real,y_pred,mode):
    
    if mode == 'm':
        mse = MSE(y_real,y_pred)
        return mse
    
    elif mode == 's':
        sse = np.sum((y_real - y_pred)**2)
        return sse
    
    elif mode == 'r2':
        return r2_score(y_real, y_pred)

## This function returns the scores for the validation set             
def model_val_scores(model, x_train_val, x_test_val, y_train_val, y_test_val, mode):
        
    model.fit(x_train_val, y_train_val)
    y_pred = model.predict(x_test_val)
    
    y_pred = np.reshape(y_pred, (y_pred.shape[0], 1)) ## Lasso shape without reshape by default: (y_pred.shape[0], ) and we want (y_pred.shape[0], 1).
    
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
    
## Load datasets
x_train = np.load('Xtrain_Regression1.npy')
y_train = np.load('Ytrain_Regression1.npy')
x_test = np.load('Xtest_Regression1.npy')

## Scalling the data in case of different types of data 
#sc_X = StandardScaler()
#x_train = sc_X.fit_transform(x_train)
#x_test = sc_X.transform(x_test)

## Normalize the data (got worse results)
#x_train = normalize(x_train)
#x_test = normalize(x_test)
#y_train = normalize(y_train)

## Evaluate training sets shape
num_rows_x, num_cols_x = x_train.shape
num_rows_y, num_cols_y = y_train.shape

## Define model and fit the training data on it 
model = Ridge(alpha=0.06)
model_f = model.fit(x_train, y_train)

## Predict for training set and test set
y_train_pred = model.predict(x_train)
y_test = model.predict(x_test)

## Saves the predicted output for the test set in .npy file
np.save('y_test', y_test)

## Evaluate regression parameters values
param_beta0 = model.intercept_
param_beta = model.coef_

## Define dataframe
df = pd.DataFrame({'y_train': y_train.tolist(), 'y_train_pred': y_train_pred.tolist()})
df.head()

## Training set scores: SSE, MSE, R^2 and MAE
print("------- Training Set Scores -------")
print('Sum of Squared Error (SSE):', scores(y_train,y_train_pred,'s'))
print('Mean Squared Error (MSE):', scores(y_train,y_train_pred,'m')) 
print('Root Mean Square Error (RMSE)', np.sqrt(scores(y_train,y_train_pred,'m')))
print('R-Squared (R2):', scores(y_train, y_train_pred, 'r2')) 
print("Mean Absolute Error (MAE):",mean_absolute_error(y_train, y_train_pred))

## Validation Phase
print("------- Validation Set Scores -------")
## KFold validation method for any type of regression 
n_splits=5
kf = KFold(n_splits, shuffle = False)
Ridge_val_scores = []
LinearR_val_scores = []
Lasso_val_scores = []
scores_list = []

for train_index, test_index in kf.split(x_train):
    x_train_val, x_test_val = x_train[train_index], x_train[test_index]
    y_train_val, y_test_val = y_train[train_index], y_train[test_index]
    Ridge_val_scores.append(model_val_scores(Ridge(alpha = 0.06), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
    LinearR_val_scores.append(model_val_scores(LinearRegression(), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
    Lasso_val_scores.append(model_val_scores(Lasso(alpha = 0.005), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
    
print("Score: SSE | K Folds Validation nº",n_splits,"splits | Ridge:", np.average(Ridge_val_scores),"| Linear:", np.average(LinearR_val_scores),"| Lasso:", np.average(Lasso_val_scores))
scores_list = [np.average(Ridge_val_scores), np.average(LinearR_val_scores), np.average(Lasso_val_scores)]
minSSE = min(scores_list)
if minSSE == scores_list[0]:
    print("-> Best is Ridge Regression:", minSSE)
elif minSSE == scores_list[1]:
    print("-> Best is Linear Regression:", minSSE)
else:
    print("-> Best is Lasso Regression:", minSSE)
    
               
## Choose the ideal Ridge Hyperparameter alpha according to the highest Cross Validation score 
ridge_cv = RidgeCV(alphas = [0.05,0.06,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17], scoring = "neg_mean_squared_error").fit(x_train,y_train)
print("Mean Squared Error (MSE) RidgeCV:", np.abs(ridge_cv.best_score_))
print("Best Ridge Hyperparameter alpha:", ridge_cv.alpha_)
print("R-Squared (R2) RidgeCV: {}".format(ridge_cv.score(x_train, y_train)))


## Calculate best split number with Cross Validation
best_score = 0
min_step = 0
for step in range(2,50):
    #print("Now testing step:",step)
    my_values = cross_val_score(Ridge(alpha=0.06), x_train, y_train, cv = step)
    my_score = np.mean(my_values)
    if my_score > best_score:
        best_score = my_score
        min_step = step
        
cv_score =  cross_val_score(Ridge(alpha=0.06), x_train, y_train, cv = min_step)  
print("Cross Validation Scores nº", min_step,"splits:", cv_score, "Average:", np.average(cv_score))



