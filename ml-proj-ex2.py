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
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LassoLars
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import balanced_accuracy_score as BACC
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import time

from openpyxl.workbook import Workbook

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
x_train = np.load('Xtrain_Regression2.npy')
y_train = np.load('Ytrain_Regression2.npy')
x_test = np.load('Xtest_Regression2.npy')

print(x_train)
print(y_train)
## Converting
"""## convert your array into a dataframe
#df = pd.DataFrame (x_train)
## save to xlsx file
#filepath = 'ex2_dataset.xlsx'
#df.to_excel(filepath, index=False)
df1 = pd.DataFrame(y_train)
filepath1 = 'ex2_ydata.xlsx'
df1.to_excel(filepath1, index=False)"""

"""Analysing the Data"""

raw_data = pd.read_csv('ex2_dataset.csv')

x_features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
y = ['y']

## Data distribution plot
for i in x_features:
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
plt.show()




