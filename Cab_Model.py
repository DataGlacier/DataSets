#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 07:11:15 2022

@author: allenayodeji
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xlrd

df = pd.read_csv('data_cleaning.csv')

date=[]
for i in df['Date of Travel']:
    xl_date = i
    datetime_date = xlrd.xldate_as_datetime(xl_date, 0)
    date_object = datetime_date.date()
    date.append(date_object)
df['Date']= date   

#To Delete a whole column
df = df.drop(['Unnamed: 0'], axis = 1)
df = df.drop(['Date of Travel'], axis = 1)
    
#To remove spaces in columns
df.columns = df.columns.str.replace(' ','_')
profit = (df.Price_Charged - df.Cost_of_Trip)
df['Profit_of_Cabs'] = profit.apply(lambda x: x if x>=0 else 0)
   


# choose relevant columns
df.columns

df_model = df[['Company', 'City', 'Date', 'Profit_of_Cabs',]]
# get dummy data
df_dum = pd.get_dummies(df_model)
# train test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('Profit_of_Cabs', axis =1)
y = df_dum.Profit_of_Cabs.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# multiple linear regression 
import statsmodels.api as sm
X_sm =  X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train))


# lasso regression
from sklearn.linear_model import Lasso
lm_l = Lasso()
lm_l.fit(X_train, y_train)
np.mean(cross_val_score(lm_l, X_train, y_train))

#Xgboost
from xgboost import XGBRegressor
xgb_model = XGBRegressor(random_state= 42)

search_space = {
    "n_estimators" : [100,200,300],
    "max_depth" : [3,6,9],
    "gamma" : [0.01, 0.1],
    "learning_rate" : [0.001, 0.01, 0.1, 1]
    
    }
# tune Models on GridSearchCV
from sklearn.model_selection import GridSearchCV
#make a GridSearchCV object
GS = GridSearchCV(estimator = xgb_model, 
                  param_grid = search_space,
                  scoring = ["r2", "neg_root_mean_squared_error"],
                  refit = "r2",
                  cv =2,
                  verbose = 4)
GS.fit(X_train, y_train)

GS.best_score_


# test ensembles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)


from sklearn.metrics import mean_absolute_error

mean_absolute_error = (y_test,tpred_lm)
mean_absolute_error = (y_test,tpred_lml)
                      