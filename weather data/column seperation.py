#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 17:06:43 2019

@author: aaa
"""

import pandas as pd



standard_levels = [1000,850,700,600,500,400,300,250,200]
df = pd.read_csv('1 January_observe.csv')

df[['time', 'day', 'month', 'year']]=df.Date.str.split(' ',expand=True)
df.drop(columns=['Date'], inplace=True)
columns = df.columns.tolist()
df = df[columns[-4:]+columns[:-4]]

df.day = pd.to_numeric(df.day)
df.year = pd.to_numeric(df.year)
error_val_index = df[(df.day==10) & (df.year==2011)].index
print(error_val_index)
#print(type(df['PRES(hPa)']))
df = df.drop(error_val_index)
#
column_name = df.columns
df[column_name] = df[column_name].apply(pd.to_numeric,errors='coerce')

column_data_types = df.dtypes

df_standard_level = df[df['PRES(hPa)'].isin(standard_levels)]