# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 09:32:49 2019

@author: iceblaze
"""

"
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import set_printoptions
from sklearn.preprocessing import StandardScaler


#load  a csv file using pandas
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

print= data.head()
