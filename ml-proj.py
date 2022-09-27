
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import balanced_accuracy_score as BACC

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
        
        elif mode == 'c':
            bacc = BACC(y_real,y_pred)
            print('The Balanced Accuracy is', bacc)
            return bacc
        
        else:
            print('You must define the mode input.')

## Load datasets
x_train = np.load('Xtrain_Regression1.npy')
y_train = np.load('Ytrain_Regression1.npy')
x_test = np.load('Xtest_Regression1.npy')

## Evaluate training sets shape
num_rows_x, num_cols_x = x_train.shape
#print ('X:', num_rows_x, num_cols_x)
num_rows_y, num_cols_y = y_train.shape
#print ('Y:', num_rows_y, num_cols_y)

## Define models
model = LinearRegression()
model_LR = model.fit(x_train, y_train)

## Define predictors
y_train_pred = model.predict(x_train)
y_pred = model.predict(x_test)
#print("y_pred:", y_pred)

## Calculate parameters values
param_beta0 = model.intercept_
param_beta = model.coef_

## Define dataframe
df = pd.DataFrame({'y_train': y_train.tolist(), 'y_train_pred': y_train_pred.tolist()})
df.head()
print(df)

## Calculate SSE and MSE
sse = np.sum((y_train_pred - y_train)**2)
print('SSE:', sse)
print('MSE:', scores(y_train,y_train_pred,'r'))

#print(param_beta)
#print(param_beta0)

#print(df['y_train'])

#sns.lmplot(data=df, x="y_train", y="y_train_pred")