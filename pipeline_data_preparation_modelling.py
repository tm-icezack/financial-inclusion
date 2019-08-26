# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:43:46 2019

@author: iceblaze
"""

#load the csv file using read_csv function of pandas library
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
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

#creating the pipeline
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model =Pipeline(estimators)

num_folds = 10
seed = 7

kfold = KFold(n_splits = num_folds, random_state = seed)


results = cross_val_score(model, X, Y, cv=kfold)
print(" mean estimated accuracy Linear_discriminant using pipeline: %f " % ( results.mean()))