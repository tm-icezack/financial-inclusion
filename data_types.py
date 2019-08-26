# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 04:29:10 2019

@author: iceblaze
"""

import pandas as pd

train = pd.read_csv( "Train_v2.csv")
test =  pd.read_csv( "Test_v2.csv")
train = train.drop('uniqueid', axis=1)


def describe_data(df):
    print("Data Types:")
    print(df.dtypes)
    print("Rows and Columns:")
    print(df.shape)
    print("Columns Names:")
    print(df.columns)
    print("Null values:")
    print(df.apply(lambda x: sum(x.isnull()) / len(df)))
describe_data(train)