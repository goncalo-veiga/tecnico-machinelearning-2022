from sklearn.preprocessing import normalize
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
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, RepeatedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor

from pyod.models.ecod import ECOD
from pyod.models.knn import KNN 

## This function returns MSE, SSE and R^2 scores
def scores(y_real,y_pred,mode):
    
    if mode == 'm':
        mse = MSE(y_real,y_pred)
        print('The Mean Square Error is', mse)
        return mse
    
    elif mode == 's':
        sse = np.sum((y_real - y_pred)**2)
        print('The Sum of Squares Error is', sse)
        return sse
            
## This function returns the scores for the validation set              
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

## Analysing the Data
raw_data = pd.read_csv('ex2_dataset.csv')
x_features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
y = ['y']

## Data Processing

x_train = normalize(x_train)
y_train = normalize(y_train)
x_train2 = np.delete(x_train,[72,34,48,63,71,96,39,5,15,20,26,41,58,93,29,51,61,73,80,99],axis=0)
y_train2 = np.delete(y_train,[72,34,48,63,71,96,39,5,15,20,26,41,58,93,29,51,61,73,80,99],axis=0)

#print(x_train)
def normalization(data):  
    
    feat_mean = np.mean(data,axis=0)
    n=0
    for row in data:
        i=0
        for value in row:
            data[n][i] = value - feat_mean[i]
            i += 1
        n += 1    
        
normalization(x_train)

""" Boxplot outliers
df_xtrain = pd.DataFrame(x_train,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'])
print(df_xtrain.head())
print(df_xtrain['x10'])
df1 = df_xtrain.drop([31,93])
print(df1['x10'])
sns.boxplot(data = df1)
plt.show()
"""

## Outlier Detection Algorithms

## Local Outlier Factor Algorithm 

model_LOF = LocalOutlierFactor(n_neighbors=20, contamination=0.2)
model_LOF_outliers = model_LOF.fit_predict(x_train)
x_train_LOF = []
y_train_LOF = []
outlier_index_LOF = []
## Remove outliers
for i in range (0, len(model_LOF_outliers)):
    if model_LOF_outliers[i] == -1:
        #print("Outlier Line:", i, x_train[i])
        #print("Outlier line:", i)
        outlier_index_LOF.append(i)
    else:
        x_train_LOF.append(x_train[i])
        y_train_LOF.append(y_train[i])

x_train_LOF = np.array(x_train_LOF)
y_train_LOF = np.array(y_train_LOF)
print("> Local Outlier Factor: Outliers detected and removed.", outlier_index_LOF)

## Isolation Forest Algorithm

model_IF = IsolationForest(contamination = 0.2,random_state=0)
model_IF_outliers = model_IF.fit_predict(x_train)
x_train_IF = []
y_train_IF = []
outlier_index_IF = []
## Remove outliers
for i in range (0, len(model_IF_outliers)):
    if model_IF_outliers[i] == -1:
        #print("Outlier Line:", i, x_train[i])
        #print("Outlier line:", i)
        outlier_index_IF.append(i)
    else:
        x_train_IF.append(x_train[i])
        y_train_IF.append(y_train[i])

x_train_IF = np.array(x_train_IF)
y_train_IF = np.array(y_train_IF)
print("> ISOLATION FOREST: Outliers detected and removed.", outlier_index_IF)

## Eliptical Envellope Algorithm

model_EE = EllipticEnvelope(contamination=0.2,random_state=0)
model_EE_outliers = model_EE.fit_predict(x_train)
x_train_EE = []
y_train_EE = []
outlier_index_EE = []
## Remove outliers
for i in range (0, len(model_EE_outliers)):
    if model_EE_outliers[i] == -1:
        #print("Outlier Line:", i, x_train[i])
        #print("Outlier line:", i)
        outlier_index_EE.append(i)
    else:
        x_train_EE.append(x_train[i])
        y_train_EE.append(y_train[i])

x_train_EE = np.array(x_train_EE)
y_train_EE = np.array(y_train_EE)
print("> ELIPTICAL ENVELLOPE: Outliers detected and removed.", outlier_index_EE)

## Find common outliers
common_outliers = {}
all_outliers = outlier_index_LOF + outlier_index_IF + outlier_index_EE
#print(" All detected outliers:", all_outliers)
for line in all_outliers:
    if line in common_outliers:
        common_outliers[line] += 1
    else:
        common_outliers[line] = 1

#print("Common outliers:",dict(sorted(common_outliers.items(), key=lambda item: item[1], reverse=True)))

## This function evaluates the different types of Regression methods with Kfolds 
def kfold_func(x_train,y_train):
    n_splits = 10
    kf = KFold(n_splits, shuffle = False)
    Ridge_val_scores = []
    LinearR_val_scores = []
    Lasso_val_scores = []
    ElasticNet_val_scores = []
    LassoLars_val_scores = []
    OMP_val_scores = []
    BayesianRidge_val_scores = []
    ARD_val_scores = []
    ridge_cv = RidgeCV(alphas = [0.5,1,1.5,1.7,1.8,1.82,1.83,1.84,1.85,1.86,1.87,1.88,1.89,1.9,1.95,2,2.05,2.1,2.16,2.17,2.18,2.19,2.2,2.21,2.22,2.23,2.25,2.27,2.3,2.4,2.5,2.6,3,3.2,3.3,3.4,
    3.5,3.6,4,5,6]).fit(x_train,y_train)
    print("alpha Ridge",ridge_cv.alpha_)
    lasso_cv = LassoCV(alphas = [1100,1200,1300,1400,1500,2000,2200,2300,2500,3000,5000,7000,10000,15000,1000000]).fit(x_train,y_train)
    print("alpha lasso",lasso_cv.alpha_)
    ElasticNet_cv = ElasticNetCV(alphas = [0.5,1,1.5,1.7,1.8,1.82,1.83,1.84,1.85,1.86,1.87,1.88,1.89,1.9,1.95,2,2.05,2.1,2.16,2.17,2.18,2.19,2.2,2.21,2.22,2.23,2.25,2.27,2.3,2.4,2.5,2.6,3,3.2,
    3.3,3.4,3.5,3.6,4,5,6,10,15,20,25,30,50,100,700]).fit(x_train,y_train)
    print("alpha ElasticNet",ElasticNet_cv.alpha_)
    LassoLars_cv = LassoLarsCV(cv = 10).fit(x_train,y_train)
    print("alpha LassoLars",LassoLars_cv.alpha_)
 
    
    for train_index, test_index in kf.split(x_train):
        x_train_val, x_test_val = x_train[train_index], x_train[test_index]
        y_train_val, y_test_val = y_train[train_index], y_train[test_index]
        Ridge_val_scores.append(model_val_scores(Ridge(alpha = ridge_cv.alpha_,random_state=0), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        LinearR_val_scores.append(model_val_scores(LinearRegression(), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        Lasso_val_scores.append(model_val_scores(Lasso(alpha=lasso_cv.alpha_,random_state=0), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        ElasticNet_val_scores.append(model_val_scores(ElasticNet(alpha=ElasticNet_cv.alpha_,random_state=0), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        LassoLars_val_scores.append(model_val_scores(LassoLars(alpha=ElasticNet_cv.alpha_,random_state=0), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        OMP_val_scores.append(model_val_scores(OrthogonalMatchingPursuit(normalize = False), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        BayesianRidge_val_scores.append(model_val_scores(BayesianRidge(), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        ARD_val_scores.append(model_val_scores(ARDRegression(alpha_1= 10, alpha_2 = 10), x_train_val, x_test_val, y_train_val, y_test_val, 's'))


    scores_list = [np.average(Ridge_val_scores), np.average(LinearR_val_scores), np.average(Lasso_val_scores),np.average(ElasticNet_val_scores),np.average(LassoLars_val_scores),np.average(OMP_val_scores),np.average(BayesianRidge_val_scores),np.average(ARD_val_scores)]
    print("Score: SSE | K Folds Validation nº",n_splits,"splits | Ridge:", scores_list[0],"| Linear:", scores_list[1],"| Lasso:", scores_list[2],"| ElasticNet:", scores_list[3]
    ,"| LassoLars:", scores_list[4],"| OMP:", scores_list[5],"| BayesianRidge:", scores_list[6],"| ARD:", scores_list[7])
    
    minSSE = min(scores_list)
    if minSSE == scores_list[0]:
        print("-> Best is Ridge Regression:", minSSE)
    elif minSSE == scores_list[1]:
        print("-> Best is Linear Regression:", minSSE)
    elif minSSE == scores_list[2]:
        print("-> Best is Lasso Regression:", minSSE)
    elif minSSE == scores_list[3]:
        print("-> Best is ElasticNet Regression:", minSSE)
    elif minSSE == scores_list[4]:
        print("-> Best is LassoLars Regression:", minSSE)
    elif minSSE == scores_list[5]:
        print("-> Best is OMP Regression:", minSSE)
    elif minSSE == scores_list[6]:
        print("-> Best is BayesianRidge Regression:", minSSE)
    elif minSSE == scores_list[7]:
        print("-> Best is ARD Regression:", minSSE)

        
print("----- Local Outlier Factor SSE-----")
kfold_func(x_train_LOF,y_train_LOF.ravel())
print("----- Isolation Forest SSE-----")
kfold_func(x_train_IF,y_train_IF.ravel())
print("----- Eliptical Envellope SSE-----")
kfold_func(x_train_EE,y_train_EE.ravel())
print("----- Removed most common outliers SSE -----")
kfold_func(x_train2,y_train2.ravel())


## This functions returns the scores for Robust Regressors such as Huber, RANSAC and Thei-sen
"""def kfold_rob_func(x_train,y_train):
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
    
    for my_alpha in range(1,1300):
        
        for train_index, test_index in kf.split(x_train):
            x_train_val, x_test_val = x_train[train_index], x_train[test_index]
            y_train_val, y_test_val = y_train[train_index], y_train[test_index]
            Huber_val_scores.append(model_val_scores(HuberRegressor(alpha= my_alpha,epsilon=1.75), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        if np.average(Huber_val_scores) < min:
            min = np.average(Huber_val_scores)
            ideal_alpha = my_alpha
    
    print("idela alpha:",ideal_alpha,"SSE:")
    print("Score: SSE | K Folds Validation nº",10,"splits | Huber:", np.average(Huber_val_scores),"| Ransac:", np.average(Ransac_val_scores),"| Theilsen:", np.average(Theilsen_val_scores))"""

"""
model_huber = OneClassSVM(kernel = 'sigmoid',nu=0.201)
model_huber_fit = model_huber.fit(x_train, y_train.ravel())
y_train_pred = model_huber.predict(x_train)
y_test = model_huber.predict(x_test)
print(y_train_pred)
count = 0
for i in range(0,len(y_train_pred)):
    if y_train_pred[i] == -1:
        count +=1
        print("i:",i)
print("counter:",count)

print("----Huber SSE----")
kfold_rob_func(normalize(x_train),normalize(y_train))
"""
