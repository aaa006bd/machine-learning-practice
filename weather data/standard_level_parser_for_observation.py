#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:01:45 2019

@author: aaa
"""
import os

import pandas as pd

directory_name='Ranchi 42701'
standard_levels = [1000,850,700,600,500,400,300,250,200]
file_names = [name for name in os.listdir(directory_name) if 'observe' in name ]
df_list = []

for file_name in file_names:
    path = os.path.join(directory_name,file_name)

    df = pd.read_csv(path,error_bad_lines=False)
        
    df[['time', 'day', 'month', 'year']]=df.Date.str.split(' ',expand=True)
    df.drop(columns=['Date'], inplace=True)
    columns = df.columns.tolist()
    df = df[columns[-4:]+columns[:-4]]
    column_data_types = df.dtypes
        
    column_name = df.columns.drop(['time','month'])
    df[column_name] = df[column_name].apply(pd.to_numeric,errors='coerce')
        
    df_standard_level = df[df['PRES(hPa)'].isin(standard_levels)]
    
    output_dir = '{} processed'.format(directory_name.split(' ')[0])
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    df_standard_level.to_csv('{}/{}_{}.csv'.format(output_dir,directory_name,file_name))
        
