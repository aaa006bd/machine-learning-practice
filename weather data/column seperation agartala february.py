#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 08:37:22 2019

@author: aaa
"""


import pandas as pd



standard_levels = [1000,850,700,600,500,400,300,250,200]
df = pd.read_csv('3 March_observe.csv')

df[['time', 'day', 'month', 'year']]=df.Date.str.split(' ',expand=True)
df.drop(columns=['Date'], inplace=True)
columns = df.columns.tolist()
df = df[columns[-4:]+columns[:-4]]
column_data_types = df.dtypes
