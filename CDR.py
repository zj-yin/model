# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:21:13 2020

@author: Server
"""

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor,RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, RANSACRegressor, ARDRegression, HuberRegressor, TheilSenRegressor,\
    SGDRegressor, PassiveAggressiveRegressor, Lasso, ElasticNet, Ridge, BayesianRidge, Lars
import os
import pandas as pd
from sklearn.model_selection import cross_val_score
import random
import math 
import numpy as np
from sklearn import metrics


data_train=pd.read_csv("train_dataset.csv")
data_test=pd.read_csv("test_dataset.csv")

feature=[]  ###feature数据集
for i in data_train.columns:
    if (i!='death_infection_rate') &(i!='country') & (i!='num') &(i!='sqrt-factor')&(i!='ICU/thousand'):
        feature.append(i)
train_feature=data_train[feature]     
train_target= data_train['death_infection_rate'] 
test_feature=data_test[feature] 




LiR=HuberRegressor()
LiR.fit(train_feature,train_target)
predictions_LiR = LiR.predict(test_feature)
print(LiR.coef_)

print(LiR.intercept_)


result1=[pd.DataFrame(data_test),pd.DataFrame(predictions_LiR)]
result1_new=pd.concat(result1,axis=1)  ###axis=1,按照列合并，=0按照行合并
result1_new.to_csv('CDR.csv', index=False)

    