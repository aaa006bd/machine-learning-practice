#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:26:31 2019

@author: aaa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import time

dataset = pd.read_csv('new_appdata10.csv')

response = dataset['enrolled']
dataset = dataset.drop(columns='enrolled')

######## DATA PREPROCESSING
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset, response, test_size=0.2, random_state=0)

train_identifier = X_train['user']
X_train.drop(columns='user', inplace=True)

test_identifier = X_test['user']
X_test.drop(columns='user', inplace=True)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))

X_train2.columns = X_train.columns.values
X_test2.columns = X_test2.columns.values

X_train2.index = X_train.index.values
X_test2.index = X_test2.index.values

X_train = X_train2
X_test = X_test2

### MODEL SELECTION
from sklearn.linear_model import LogisticRegression
#penalty l1 regularization is important for mobile screen coz it penalizes the correlated screens
classifier = LogisticRegression(random_state=0, penalty='l1')
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall= recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

from sklearn.model_selection import cross_val_score
accuracies_cross_val = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)


final_results = pd.concat([y_test, test_identifier], axis=1).dropna()
final_results['predicted_results']=y_pred
final_results[['user','enrolled','predicted_results']].reset_index(drop=True)
