# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 07:36:50 2019

@author: iceblaze
"""
import pandas as pd
data = pd.read_csv( 'submission_06.csv')

data['bank_account'] = data['bank_account'].map({'Yes':1,'No':0})

data.to_csv('submission_008.csv', index=False)

