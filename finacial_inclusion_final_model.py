# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 03:02:56 2019

@author: iceblaze
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from pickle import dump

#load the csv file using read_csv fuction of pandas libary
data = pd.read_csv( "Train_v2.csv")

#assinging dependent variable 
Y = data.iloc[:,3].values
# assinging independent varible
X = data.drop(columns=['bank_account','uniqueid', 'household_size','age_of_respondent' ]).astype(str)
#seperating the non-categorical features
z = data.iloc[:,[6,7]]
labelencoder_x = LabelEncoder()
X = X.apply(lambda col: labelencoder_x.fit_transform(col))
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)
#avoiding the dummy variable trap
X = X[:,1:]
X = pd.DataFrame(data=X)
X = X.join(z).values

######## DO REQUIRE SUMMAIZATION, EVALUTIONS ANS OPTIMIZATION HERE ########
 
model = LogisticRegression(solver='liblinear')
model.fit(X,Y)
 
#SAVE THIS MODEL
filename = ' final_inclusion.sav'
dump(model,open(filename,'wb'))