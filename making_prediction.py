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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

train = pd.read_csv( "Train_v2.csv")
test =  pd.read_csv( "Test_v2.csv")



numeric_features = train.select_dtypes(include=['int64','float64']).columns
categorical_features = train.select_dtypes(include=['object']).drop(['bank_account'], axis=1).columns

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scalar', StandardScaler())])

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

#concate both the numeric and categorical transformer into an object called preprocessor

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),('cat',categorical_transformer, categorical_features)])

#training the model
X = train.drop( 'bank_account', axis=1)
Y = train['bank_account']
test_size = 0.33
seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#fitting and predicting
lr = Pipeline(steps=[('preprocessor',preprocessor), ('classifier', LogisticRegression(solver='liblinear'))])

lr.fit(X_train, Y_train)
print("model score:%.3f"% lr.score(X_test, Y_test))


#making a prediction
test_no_id = test.drop('uniqueid', axis=1)
test_predictions = lr.predict(test_no_id)
uniqueid = test['uniqueid'] +' x '+ test['country']
 
submission_df_1 = pd.DataFrame({"uniqueid": uniqueid, "bank_account": test_predictions})

submission_df_1.to_csv('submission_06.csv', index=False)



