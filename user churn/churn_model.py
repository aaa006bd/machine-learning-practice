#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:05:21 2019

@author: aaa
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random

dataset = pd.read_csv('new_churn_data.csv')

user_identifier = dataset['user']

dataset.drop(columns=['user'], inplace=True)


#one hot encoding
dataset=pd.get_dummies(dataset)
columns=dataset.columns.tolist()
dataset.drop(columns=['housing_na','payment_type_na','zodiac_sign_na'],inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns='churn'),
                                                    dataset['churn'],test_size=0.25, random_state=0)

#balancing the training set

pos_index = y_train[y_train.values==1].index
neg_index = y_train[y_train.values==0].index

if len(pos_index)> len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    higher=neg_index
    lower=pos_index

random.seed(0)
higher = np.random.choice(higher, size = len(lower))
lower = np.asarray(lower)

new_indexes = np.concatenate((lower,higher))

X_train = X_train.loc[new_indexes,]
y_train = y_train[new_indexes]

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2= pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))

X_train2.columns = X_train.columns.values
X_train2.index = X_train.index.values

X_test2.columns = X_test.columns.values
X_test2.index = X_test.index.values

X_train = X_train2
X_test = X_test2


## Model building ##

#model fitting 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

#testing model 
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

#cm plot
df_cm = pd.DataFrame(cm, index = (0,1), columns = (0,1))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')

#k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv = 10)

#analyzing coefficient
feature_frame = pd.concat([pd.DataFrame(X_train.columns, columns=['features']),
           pd.DataFrame(np.transpose(classifier.coef_),columns=['coef'])],
           axis=1)


#feature selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
rfe = RFE(classifier, 20)
rfe = rfe.fit(X_train, y_train)

selected_columns = X_train.columns[rfe.support_].tolist()
selected_features_ranking = pd.concat([pd.DataFrame(X_train.columns, columns=['features']),
           pd.DataFrame(np.transpose(rfe.ranking_),columns=['ranking'])],
           axis=1)
#building model with new features
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train[X_train.columns[rfe.support_]],y_train)

y_pred=classifier.predict(X_test[X_test.columns[rfe.support_]])

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

#cm plot
df_cm = pd.DataFrame(cm, index = (0,1), columns = (0,1))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')











