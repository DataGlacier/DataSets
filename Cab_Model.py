#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 07:11:15 2022

@author: allenayodeji
"""

import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
   

# saving a holdout test set of 100 rows
holdout = df.iloc[-3000:, :]
# saving as a json to test later
holdout.to_json("holdout_test.json", orient="records")

# the non-holdout data is train data
train = df.iloc[:3000, :]

def pre_process_data(train):
    # dropping customerID column. Since it is unique to each customer,
    # it is not useful to train on.
    train.drop("Customer_ID", axis=1, inplace=True)
    train.drop("Transaction_ID", axis=1, inplace=True)
    train.drop("Payment_Mode", axis=1, inplace=True)
    
    categorical_columns=  ['Company', 'City', 'KM_Travelled', 'Price_Charged',
           'Cost_of_Trip','Gender', 'Age',
           'Income_(USD/Month)', 'Date', 'Profit_of_Cabs']
    
    # converting all the categorical columns to numeric
    col_mapper = {}
    class_names_mapper = {}
    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(train.loc[:, col])
        class_names = le.classes_
        train.loc[:, col] = le.transform(train.loc[:, col])
        # saving encoder for each column to be able to inverse-transform later
        col_mapper.update({col: le})
        class_names_mapper.update({col: class_names})
        
        # handling issue where numeric columns have blank rows
    train.replace(" ", "0", inplace=True)


    return train, col_mapper, class_names


# applying pre-process function
processed_train, col_mapper, class_names_mapper = pre_process_data(train)  


# splitting into X and Y
x_train = processed_train.drop("Profit_of_Cabs", axis=1)
y_train = processed_train.loc[:, "Profit_of_Cabs"]


# training out-of-the-box Logistic Regression
model = LogisticRegression()
model.fit(x_train, y_train)

# getting predictions
predictions = model.predict(x_train)
accuracy = accuracy_score(y_train, predictions)
# checking accuracy of predictions
print(accuracy)

# pickling mdl
pickler = open("Cab_prediction_model.pkl", "wb")
pickle.dump(model, pickler)
pickler.close()

# pickling le dict
pickler = open("Cab_prediction_label_encoders.pkl", "wb")
pickle.dump(col_mapper, pickler)
pickler.close()

# pickling class names dict
pickler = open("Cab_prediction_class_names.pkl", "wb")
pickle.dump(class_names_mapper, pickler)
pickler.close()

