#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:58:11 2019

@author: aaa
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('churn_data.csv')
columns = dataset.columns.tolist()
stat = dataset.describe()

##DATA CELANING
#find out which columns contain na value
nan_columns = dataset.isna().any()

#find out how many nans in each column
nan_count = dataset.isna().sum()

#remove rows where age is null
dataset =  dataset[pd.notnull(dataset['age'])]

dataset.drop(columns=['credit_score','rewards_earned'], inplace=True)


##HISTOGRAMS

dataset_numeric = dataset.drop(columns = ['user','churn'])

#fig = plt.figure(figsize=(5,12))
#plt.suptitle('histogram of numerical columns', fontsize = 20)
#for i in range(1, dataset_numeric.shape[1]+1):
#    plt.subplot(6,5, i)
#    f=plt.gca()
#    f.axes.get_yaxis().set_visible(False)
#    f.set_title(dataset_numeric.columns.values[i-1])
#    
#    vals = np.size(dataset_numeric.iloc[:, i-1].unique())
#    
#    plt.hist(dataset_numeric.iloc[:, i-1], bins=vals)
#             
#plt.tight_layout(rect=[0,0.03,1,0.95])

dataset_binary=dataset[['housing', 'is_referred', 'app_downloaded',
                        'web_user', 'app_web_user', 'ios_user','android_user',
                        'registered_phones','payment_type','waiting_4_loan','cancelled_loan',
                        'received_loan','rejected_loan', 'zodiac_sign',
                        'left_for_two_month_plus','left_for_one_month','is_referred']]

##PIECHART DISTRIBUTIONS

#fig = plt.figure(figsize=(15,12))
#plt.suptitle('Pie chart of numerical columns', fontsize = 20)
#for i in range(1, dataset_binary.shape[1]+1):
#    plt.subplot(6,3, i)
#    f=plt.gca()
#    f.axes.get_yaxis().set_visible(False)
#    f.set_title(dataset_binary.columns.values[i-1])
#    values = dataset_binary.iloc[:, i-1].value_counts(normalize = True).values
#    index = dataset_binary.iloc[:, i-1].value_counts(normalize = True).index
#
#         
#    plt.pie(values, labels=index, autopct='%1.1f%%')
#             
#plt.tight_layout(rect=[0,0.05,1,0.90])  

##EXPLORING UNEVEN FEATURES

dataset.drop(columns=['user','churn','housing',
                      'payment_type', 'zodiac_sign']).corrwith(dataset.churn).plot.bar(
    figsize=(20,10), title='correlation with response variable', fontsize=10, rot = 45, grid=True)

sn.set(style='white', font_scale=1)
corr = dataset.drop(columns=['user','churn']).corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(18,15))
f.suptitle('correlation', fontsize=10)

cmap = sn.diverging_palette(220, 10, as_cmap=True)

sn.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True,
           lineWidths =0.5, cbar_kws={'shrink':0.5})
 

dataset.drop(columns=['app_web_user','ios_user'],inplace = True)
dataset.to_csv('new_churn_data.csv', index=False)
