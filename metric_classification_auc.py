# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:35:12 2019

@author: iceblaze
"""

#load the csv file using read_csv function of pandas library
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
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



kfold = KFold(n_splits=10, random_state = 7)
model = LogisticRegression(solver='liblinear')
scoring = 'roc_auc'

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f) " % (results.mean(), results.std()))


