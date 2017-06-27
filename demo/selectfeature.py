# -*- coding: utf-8 -*-
"""
Created on Tue May 16 08:40:00 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
train_data = pd.read_csv(r'E:\tencent\input\train.txt')
print(1)
test_data = pd.read_csv(r'E:\tencent\input\test.txt')
print(2) 
#处理空值
test_data = test_data.fillna(value=0)
train_data = train_data.fillna(value=0) 
names = train_data.columns[1:]
#获取训练集L
#trains = train_data.drop('f1')
labels = train_data.iloc[:,:1]
labels = np.ravel(labels) 
trains = train_data.iloc[:,3:]
trains = np.array(trains)
rf = RandomForestRegressor()
rf.fit(trains,labels)
print('Features sorted by their score:')
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))
