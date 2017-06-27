# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:19:37 2017

@author: Administrator
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy as sp
import time 
from sklearn.model_selection import train_test_split

now = time.time() 
train_data = pd.read_pickle(r'E:\tencent\input\nofeature\trainresult')
print(1)
test_data = pd.read_pickle(r'E:\tencent\input\nofeature\test')
print(2) 
#处理空值
test_data = test_data.fillna(value=0)
train_data = train_data.fillna(value=0) 
#获取训练集L

#trains = train_data.drop('f1')
labels = train_data.iloc[:,:1].values
trains = train_data.iloc[:,3:]
#trains = trains.drop('conversionTime',axis=1).values  
X_train, X_test, y_train, y_test = train_test_split(trains,labels,test_size = 0.2,random_state = 42)

clf = LinearRegression(n_jobs=4)
clf.fit(X_train,y_train)
clf.coef_