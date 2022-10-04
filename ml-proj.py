import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import balanced_accuracy_score as BACC
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

## Scores between the y real and y predicted
def scores(y_real,y_pred,mode):
    ###y_real - ground truth vector 
    ###y_pred - vector of predictions, must have the same shape as y_real
    ###mode   - if evaluating regression ('r') or classification ('c')
    
    if y_real.shape != y_pred.shape:
        print('confirm that both of your inputs have the same shape')
    else:
        if mode == 'r':
            mse = MSE(y_real,y_pred)
            print('The Mean Square Error is', mse)
            return mse

        
        elif mode == 's':
            sse = np.sum((y_real - y_pred)**2)
            print('The Sum of Squares Error is', sse)
            return sse
        
        else:
            print('You must define the mode input.')

## Validation Scores inside the validation set             
def model_val_scores(model, x_train_val, x_test_val, y_train_val, y_test_val, mode):
    
    model.fit(x_train_val, y_train_val)
    y_pred = model.predict(x_test_val)
    
    if mode == 'r':
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
x_train = np.load('Xtrain_Regression1.npy')
y_train = np.load('Ytrain_Regression1.npy')
x_test = np.load('Xtest_Regression1.npy')

## Scalling the data in case of different types of data 
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

## Evaluate training sets shape
num_rows_x, num_cols_x = x_train.shape
num_rows_y, num_cols_y = y_train.shape

## Define models and fitting the training data on it 
model = Ridge(alpha=0.06)
model_LR = model.fit(x_train, y_train)

## Define predictors for training set and test ste
y_train_pred = model.predict(x_train)
y_pred = model.predict(x_test)

## Evaluate regression parameters values
param_beta0 = model.intercept_
param_beta = model.coef_
print("Beta0:",param_beta0,"beta:",param_beta)

## Define dataframe
df = pd.DataFrame({'y_train': y_train.tolist(), 'y_train_pred': y_train_pred.tolist()})
df.head()
print(df)

## Calculate SSE, MSE, R^2 and MAE
print("-------Without Validation Results-------")
print('SSE:', scores(y_train,y_train_pred,'s'))
print('MSE:', scores(y_train,y_train_pred,'r'))
R2_score = r2_score(y_train, y_train_pred) #regression R^2 score
print('R^2:', R2_score) 
MAE = mean_absolute_error(y_train, y_train_pred)
print("Mean Absolute Error:", MAE)

"""Validation"""
print("-------Validation Set Results-------")
## Define KFold validation method 
kf = KFold(n_splits=4)
val_scores = []

## KFold Validation for any type of regression and mode defined
for train_index, test_index in kf.split(x_train):
    #print("TRAIN:", train_index, "\nTEST:", test_index)
    x_train_val, x_test_val = x_train[train_index], x_train[test_index]
    y_train_val, y_test_val = y_train[train_index], y_train[test_index]
    val_scores.append(model_val_scores(Ridge(alpha=0.06), x_train_val, x_test_val, y_train_val, y_test_val, 's'))


print("K Folds Validation Scores method:", val_scores)
print("SSE Score:",np.mean(val_scores))

## Calculate ideal alpha
ridge_cv = RidgeCV(alphas = [0.05,0.06,0.08,0.09,0.1]).fit(x_train,y_train)

## Calculate best split number
min_score = 0
min_step = 0
for step in range(2,50):
    #print("Now testing step:",step)
    my_values = cross_val_score(Ridge(alpha=0.06), x_train, y_train, cv = step)
    my_score = np.mean(my_values)
    if my_score > min_score:
        min_score = my_score
        min_step = step

#print("Min Score:", min_score, "split:", min_step)

print("Cross Validation Scores method:", cross_val_score(Ridge(alpha=0.06), x_train, y_train, cv = min_step))

#score
print("The train score for ridge model is {}".format(ridge_cv.score(x_train, y_train)))
print("Best alpha:", ridge_cv.alpha_)



