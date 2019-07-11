#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:10:03 2019

@author: aaa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from dateutil import parser

dataset = pd.read_csv('data/appdata10.csv')
stat = dataset.describe()
dataset['hour']=dataset.hour.str.slice(1,3).astype(int) 

column_list = ['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled']
dataset_updated = dataset.copy().drop(columns=column_list)

## CORRELATION MATRIX VIA HEAT MAP

#plt.suptitle('histogram of numerical columns', fontsize = 20)

#Histogram plotting
#for i in range(1, dataset_updated.shape[1]+1):
#    plt.subplot(3,3, i)
#    f=plt.gca()
#    f.set_title(dataset_updated.columns.values[i-1])
#    
#    vals = np.size(dataset_updated.iloc[i-1].unique())
#    plt.hist(dataset_updated.iloc[:, i-1], bins=vals, color = '#3F5D7D')
             
#dataset_updated.plot.hist(bins=12, alpha=0.5)

#correlation with response
#dataset_updated.corrwith(dataset.enrolled).plot.bar(figsize = (20,10),
#                         title = 'correlation with response variable',
#                         fontsize = 15, rot = 45,
#                         grid = True)
#
#sn.set(style='white', font_scale=2)
#corr = dataset_updated.corr()
#
#mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
#
#f, ax = plt.subplots(figsize=(18,15))
#f.suptitle('correlation', fontsize=40)
#
#cmap = sn.diverging_palette(220, 10, as_cmap=True)
#
#sn.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True,
#           lineWidths =0.5, cbar_kws={'shrink':0.5})

## FINDING DIFFERENCE BETWEEN TIMES
dataset['first_open'] = [parser.parse(row_data) for row_data in dataset['first_open']]
dataset['enrolled_date'] = [parser.parse(row_data) if isinstance(row_data, str) else row_data  for row_data in dataset['first_open']]

dataset['difference'] = (dataset.enrolled_date - dataset.first_open).astype('timedelta64[h]')

plt.hist(dataset['difference'].dropna(), color='#3F5D7D')
plt.title('distribution of time since enrolled')
plt.show()

plt.hist(dataset['difference'].dropna(), color='#3F5D7D', range=[0,100])
plt.title('distribution of time since enrolled')
plt.show()

## FEATTURE ENGINEERING
#cut of response. we will only consider people who enrolled within 48 hours 
#we used difference of time and histogram to see in what time range most of the enrollment occured 
dataset.loc[dataset.difference > 48, 'enrolled'] = 0
dataset.drop(columns = ['difference', 'enrolled_date', 'first_open'], inplace=True)


## FORMATTING THE SCREEN_LIST FIELD
top_screens = pd.read_csv('data/top_screens.csv').top_screens.values

dataset['screen_list'] = dataset.screen_list.astype(str)+','

for sc in top_screens:
     dataset[sc]=dataset.screen_list.str.contains(sc).astype(int)
     dataset['screen_list']=dataset.screen_list.str.replace(sc+',', '')

dataset['other']=dataset.screen_list.str.count(',')
dataset.drop(columns=['screen_list'], inplace=True)

## FUNNELS

saving_screens = ['Saving1',
                  'Saving2',
                  'Saving2Amount',
                  'Saving4',
                  'Saving5',
                  'Saving6',
                  'Saving7',
                  'Saving8',
                  'Saving9',
                  'Saving10',]

dataset["savingsCount"]= dataset[saving_screens].sum(axis = 1)
dataset.drop(columns = saving_screens, inplace=True)

cm_screens = ['Credit1',
              'Credit2',
              'Credit3',
              'Credit3Container',
              'Credit3Dashboard']

dataset['CMCount']=dataset[cm_screens].sum(axis=1)
dataset.drop(columns = cm_screens, inplace=True)

cc_screens = ['CC1',
              'CC1Category',
              'CC3']

dataset['CCCount']=dataset[cc_screens].sum(axis=1)
dataset.drop(columns = cc_screens, inplace=True)

loan_screens = ['Loan',
                'Loan2',
                'Loan3',
                'Loan4',]

dataset['LoansCount']=dataset[loan_screens].sum(axis=1)
dataset.drop(columns = loan_screens, inplace=True)

dataset.to_csv('new_appdata10.csv')