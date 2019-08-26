# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:37:46 2019

@author: iceblaze
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

test_size = 0.33

seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=test_size, random_state=seed)
model = LogisticRegression(solver='liblinear')

model.fit(X_train, Y_train)
predicted = model.predict(X_test)

report = classification_report(Y_test, predicted)
print(report)
