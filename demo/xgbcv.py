# -*- coding: utf-8 -*-
"""
Created on Sun May 21 12:27:19 2017

@author: Administrator
"""

import pandas as pd
import pandas as pd
import xgboost as xgb
import numpy as np
import time
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn_pandas import DataFrameMapper
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split

now = time.time() 
train_data = pd.read_pickle(r'E:\tencent\input\noonehot\train')
print(1)
test_data = pd.read_pickle(r'E:\tencent\input\noonehot\test')
print(2) 
#处理空值
test_data = test_data.fillna(value=0)
train_data = train_data.fillna(value=0) 
#获取训练集L

#trains = train_data.drop('f1')
labels = train_data.iloc[:,:1].values
trains = train_data.iloc[:,3:]
train_data['appCategory_1'] = train_data['appCategory'].map(lambda x: round(x / 100))   # 列处理
train_data['appCategory_2'] = train_data['appCategory'].map(lambda x: x % 100)
train_data['home_province'] = train_data['hometown'].map(lambda x: round(x / 100))
train_data['home_city'] = train_data['hometown'].map(lambda x: x % 100)
train_data['live_province'] = train_data['residence'].map(lambda x: round(x / 100))
train_data['live_city'] = train_data['residence'].map(lambda x: x % 100)
#处理test
test_data['appCategory_1'] = test_data['appCategory'].map(lambda x: round(x / 100))   # 列处理
test_data['appCategory_2'] = test_data['appCategory'].map(lambda x: x % 100)
test_data['home_province'] = test_data['hometown'].map(lambda x: round(x / 100))
test_data['home_city'] = test_data['hometown'].map(lambda x: x % 100)
test_data['live_province'] = test_data['residence'].map(lambda x: round(x / 100))
test_data['live_city'] = test_data['residence'].map(lambda x: x % 100)
#trains = trains.drop('conversionTime',axis=1).values  
#X_train, X_test, y_train, y_test = train_test_split(trains,labels,test_size = 0.2,random_state = 42)

#获取测试集
tests = test_data.iloc[:,3:] 
print(4)
#设置参数
parms = {
    'booster': 'gbtree',
    # 这里二分类，是一个二类的问题，因此采用了binary:logistic分类器，
    'objective': 'binary:logistic',
    'gamma': 0.1,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'scale_pos_weight':1,
    'lambda':2,
    'max_depth': 10,  # 构建树的深度 [1:] 最佳
    # 'lambda':450,  # L2 正则项权重
    'subsample': 0.8,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_bytree': 0.7,  # 构建树树时的采样比率 (0:1]
   # 'min_child_weight':8, # 节点的最少特征数 
    'silent': 1,
    'eta': 0.01,  # 如同学习率
    'seed': 780,
    'nthread': 7,
    'max_delta_step':6,
    'min_child_weight':4, #最佳
    'eval_metric':'logloss',
   # 'n_estimators':1000
} 

#
plst = list(parms.items()) 
xgTrain = xgb.DMatrix(train_data,label=labels)
print(5)
xgb.cv(parms,train_data,num_boost_round=200,nfold=5,metrics='logloss',show_stdv=False)
