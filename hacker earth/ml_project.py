#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:17:58 2019

@author: aaa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm

plt.rcParams['figure.figsize'] = (10.0, 8.0)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#miss = train.isnull().sum()/len(train)
#miss = miss[miss>0]
#miss.sort_values(inplace=True)
#
#miss_frame = miss.to_frame()
#miss_frame.columns = ['count']
#miss_frame.index.names = ['Name']
#miss_frame['Name'] = miss.index
#
#sns.set(style='darkgrid', color_codes=True)
#sns.barplot(x='Name', y='count', data=miss_frame)
#
#plt.xticks(rotation=90)

ax = sns.distplot(train['SalePrice'])

skewdenss_salePrice = train['SalePrice'].skew()

plt.show()