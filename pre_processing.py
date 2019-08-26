# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 04:29:10 2019

@author: iceblaze
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

train = pd.read_csv( "Train_v2.csv")
test =  pd.read_csv( "Test_v2.csv")



numeric_features = train.select_dtypes(include=['int64','float64']).columns
categorical_features = train.select_dtypes(include=['object']).drop(['uniqueid'], axis=1).columns

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scalar', StandardScaler())])

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

#concate both the numeric and categorical transformer into an object called preprocessor

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),('cat',categorical_transformer, categorical_features)])
