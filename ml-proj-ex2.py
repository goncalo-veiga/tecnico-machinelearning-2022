"""
Authors: 
96738 - Gonçalo Veiga 
96216 - Gonçalo Galante
"""
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
from sklearn.linear_model import Lars
from sklearn.linear_model import LarsCV
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import TheilSenRegressor
from sklearn.preprocessing import RobustScaler
from pyod.models.ecod import ECOD
from pyod.models.knn import KNN 
pd.set_option('display.max_rows', None)

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

# This function standardizes our features according to the mean and std 
def standardization(data):  
    feat_mean = np.mean(data,axis=0)
    feat_std = np.std(data, axis = 0)
    n=0
    for row in data:
        i=0
        for value in row:
            data[n][i] = (value - feat_mean[i])/feat_std[i]
            i += 1
        n += 1

## Load datasets
x_train = np.load('Xtrain_Regression2.npy')
y_train = np.load('Ytrain_Regression2.npy')
x_test = np.load('Xtest_Regression2.npy')

# Boxplot outliers
df_xtrain = pd.DataFrame(x_train,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'])
df_xtest = pd.DataFrame(x_test,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'])
print(df_xtrain.head())
df1 = df_xtrain.drop([46,6,59,63,13,17,22])
sns.boxplot(data = df_xtrain)
plt.show()

## Removing outliers found by boxplot/IQR
x_train2 = np.delete(x_train,[46,6,59,63,13,17,22],axis=0)
y_train2 = np.delete(y_train,[46,6,59,63,13,17,22],axis=0)

## Only Isolation Forest Algorithm
model_IF2 = IsolationForest(contamination = 0.2,random_state=0)
model_IF_outliers2 = model_IF2.fit_predict(x_train)
x_train_IF2, y_train_IF2, outlier_index_IF2 = [],[],[]
## Remove outliers
for i in range (0, len(model_IF_outliers2)):
    if model_IF_outliers2[i] == -1:
        outlier_index_IF2.append(i)
    else:
        x_train_IF2.append(x_train[i])
        y_train_IF2.append(y_train[i])

x_train_IF2 = np.array(x_train_IF2)
y_train_IF2 = np.array(y_train_IF2)
print("> Isolation Forest: Outliers detected and removed.", outlier_index_IF2)

## Local Outlier Factor Algorithm (IQR outliers removed) 
model_LOF = LocalOutlierFactor(n_neighbors=20, contamination=0.13)
model_LOF_outliers = model_LOF.fit_predict(x_train2)
x_train_LOF,y_train_LOF,outlier_index_LOF = [],[],[]
## Remove outliers: LOF
for i in range (0, len(model_LOF_outliers)):
    if model_LOF_outliers[i] == -1:
        outlier_index_LOF.append(i)
    else:
        x_train_LOF.append(x_train2[i])
        y_train_LOF.append(y_train2[i])
x_train_LOF = np.array(x_train_LOF)
y_train_LOF = np.array(y_train_LOF)
print("> Local Outlier Factor: Outliers detected and removed.", outlier_index_LOF)

## Eliptical Envellope Algorithm (IQR outliers removed) 
model_EE = EllipticEnvelope(contamination=0.13,random_state=0)
model_EE_outliers = model_EE.fit_predict(x_train2)
x_train_EE,y_train_EE,outlier_index_EE = [],[],[]
## Remove outliers
for i in range (0, len(model_EE_outliers)):
    if model_EE_outliers[i] == -1:
        outlier_index_EE.append(i)
    else:
        x_train_EE.append(x_train2[i])
        y_train_EE.append(y_train2[i])

x_train_EE = np.array(x_train_EE)
y_train_EE = np.array(y_train_EE)
print("> Eliptical Envellope: Outliers detected and removed.", outlier_index_EE)

## Isolation Forest (IQR outliers removed) 
model_IF = IsolationForest(contamination = 0.13,random_state=0)
model_IF_outliers = model_IF.fit_predict(x_train2)
x_train_IF,y_train_IF,outlier_index_IF = [],[],[]
## Remove outliers: IF
for i in range (0, len(model_IF_outliers)):
    if model_IF_outliers[i] == -1:
        outlier_index_IF.append(i)
    else:
        x_train_IF.append(x_train2[i])
        y_train_IF.append(y_train2[i])

x_train_IF = np.array(x_train_IF)
y_train_IF = np.array(y_train_IF)
print("> Isolation Forest: Outliers detected and removed.", outlier_index_IF)

## Find most common outliers
common_outliers = {}
all_outliers = outlier_index_LOF + outlier_index_IF + outlier_index_EE
#print(" All detected outliers:", all_outliers)
for line in all_outliers:
    if line in common_outliers:
        common_outliers[line] += 1
    else:
        common_outliers[line] = 1

print("Common outliers:",dict(sorted(common_outliers.items(), key=lambda item: item[1], reverse=True)))

## This function evaluates the different types of Regression methods with Kfolds 
def kfold_func(x_train,y_train,n_splits):
    #transformer = RobustScaler().fit(x_train)
    #scaler = MinMaxScaler().fit(x_train)
    #x_train = scaler.transform(x_train)

    kf = KFold(n_splits)
    (Ridge_val_scores, LinearR_val_scores, Lasso_val_scores, ElasticNet_val_scores, LassoLars_val_scores, OMP_val_scores,
     BayesianRidge_val_scores, ARD_val_scores, Lars_val_scores)  = [],[],[],[],[],[],[],[],[]

    ridge_cv = RidgeCV(cv=n_splits,scoring='neg_mean_squared_error').fit(x_train,y_train)
    print("alpha Ridge:",ridge_cv.alpha_)
    lasso_cv = LassoCV(cv=n_splits).fit(x_train,y_train)
    print("alpha Lasso:",lasso_cv.alpha_)
    ElasticNet_cv = ElasticNetCV(cv=n_splits, l1_ratio = 1).fit(x_train,y_train)
    print("alpha ElasticNet:",ElasticNet_cv.alpha_)
    LassoLars_cv = LassoLarsCV(cv = n_splits).fit(x_train,y_train)
    print("alpha LassoLars:",LassoLars_cv.alpha_)
    Lars_cv = LarsCV(cv = n_splits).fit(x_train,y_train)
    print("alpha Lars:",Lars_cv.alpha_)

    for train_index, test_index in kf.split(x_train):
        x_train_val, x_test_val = x_train[train_index], x_train[test_index]
        y_train_val, y_test_val = y_train[train_index], y_train[test_index]
        Ridge_val_scores.append(model_val_scores(Ridge(alpha = ridge_cv.alpha_,random_state=0), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        LinearR_val_scores.append(model_val_scores(LinearRegression(), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        Lasso_val_scores.append(model_val_scores(Lasso(alpha=lasso_cv.alpha_,random_state=0), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        ElasticNet_val_scores.append(model_val_scores(ElasticNet(alpha=ElasticNet_cv.alpha_,random_state=0, l1_ratio=1), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        LassoLars_val_scores.append(model_val_scores(LassoLars(alpha=LassoLars_cv.alpha_,random_state=0), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        OMP_val_scores.append(model_val_scores(OrthogonalMatchingPursuit(normalize = False), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        BayesianRidge_val_scores.append(model_val_scores(BayesianRidge(), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        ARD_val_scores.append(model_val_scores(ARDRegression(alpha_1= 20, alpha_2 = 20), x_train_val, x_test_val, y_train_val, y_test_val, 's'))
        Lars_val_scores.append(model_val_scores(Lars(), x_train_val, x_test_val, y_train_val, y_test_val, 's'))

    scores_list = [np.average(Ridge_val_scores), np.average(LinearR_val_scores), np.average(Lasso_val_scores),np.average(ElasticNet_val_scores),
    np.average(LassoLars_val_scores),np.average(OMP_val_scores),np.average(BayesianRidge_val_scores),np.average(ARD_val_scores),np.average(Lars_val_scores)]
    print("Score: SSE | K Folds Validation nº",n_splits,"splits | Ridge:", scores_list[0],"| Linear:", scores_list[1],"| Lasso:", scores_list[2],"| ElasticNet:", scores_list[3]
    ,"| LassoLars:", scores_list[4],"| OMP:", scores_list[5],"| BayesianRidge:", scores_list[6],"| ARD:", scores_list[7],"| Lars:", scores_list[8])
    
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
    elif minSSE == scores_list[8]:
        print("-> Best is Lars Regression:", minSSE)

print("----- With outliers SSE-----")
kfold_func(x_train,y_train.ravel(),80)
print("----- Only Isolation Forest SSE-----")
kfold_func(x_train_IF2,y_train_IF2.ravel(),80)
print("----- Pre-removed IQR outliers and Local Outlier Factor SSE-----")
kfold_func(x_train_LOF,y_train_LOF.ravel(),80)
print("----- Pre-removed IQR outliers and Eliptical Envellope SSE-----")
kfold_func(x_train_EE,y_train_EE.ravel(),80)
print("----- Removed most common outliers SSE -----")
kfold_func(x_train2,y_train2.ravel(),80)

## Gives the best result
print("----- Pre-removed IQR outliers and Isolation Forest SSE-----")
kfold_func(x_train_IF,y_train_IF.ravel(),80)


## Define model and fit the training data on it 
our_model = Lasso(alpha=0.31036887748972686,random_state=0)
model_f = our_model.fit(x_train_IF, y_train_IF)

## Predict for the test set
y_test = our_model.predict(x_test)
y_test = np.reshape(y_test,(1000,1))

## Saves the predicted output for the test set in .npy file
np.save('y_test2', y_test)
