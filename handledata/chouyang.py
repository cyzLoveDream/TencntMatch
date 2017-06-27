# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:57:36 2017

@author: Administrator

"""
import xgboost as xgb
import numpy as np
import time 
from collections import Counter
import pandas as pd
from sklearn.datasets import make_classification
from imblearn.ensemble import BalanceCascade
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn_pandas import DataFrameMapper


now = time.time()
testData = pd.read_pickle(r'E:\tencent\input\noonehot\train')
testData = testData.fillna(0)
testData_x = testData.iloc[:,3:]
testData_y = testData.iloc[:,:1]


#x,y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

print('Original dataset shape {}'.format(Counter(testData_y)))   
#采样
bc = BalanceCascade(random_state=42)
X_res, y_res = bc.fit_sample(testData_x, testData_y)
#print(X_res)
#print(y_res)
print('Resampled dataset shape {}'.format(Counter(y_res[0])))
print(time.time()-now)
#获取训练集L
#trains = train_data.drop('f1')
labels = pd.DataFrame(y_res[0])
trains = pd.DataFrame(X_res[0])
#trains.columns = ['creativeID', 'userID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'advertiserID', 'appID', 'appPlatform', 'sitesetID', 'positionType']
'''
mapper =DataFrameMapper([
            ('education',LabelBinarizer()),
            (['positionID','connectionType','telecomsOperator','gender','education','marriageStatus',
              'haveBaby','appPlatform','sitesetID','positionType'],OneHotEncoder()),
           # ('hometown',[FunctionTransformer(lambda x: x%100),MultiLabelBinarizer()]), 
            (['creativeID'],[StandardScaler(),PolynomialFeatures(2)]),
             
            (['age','userID'],[MinMaxScaler(),StandardScaler(),PolynomialFeatures(2)])
            ],None)
         
trains=mapper.fit_transform(trains)
 '''
#X_res.to_csv(r'E:\tencent\resample.csv')
#trains = trains.drop('conversionTime',axis=1).values
                        
#获取测试集
test_data = pd.read_csv(r'E:\tencent\pre\testresult.csv')
tests = test_data.iloc[:,3:]
'''
mapper =DataFrameMapper([
            ('education',LabelBinarizer()),
            (['positionID','connectionType','telecomsOperator','gender','education','marriageStatus',
              'haveBaby','appPlatform','sitesetID','positionType'],OneHotEncoder()),
           # ('hometown',[FunctionTransformer(lambda x: x%100),MultiLabelBinarizer()]), 
            (['creativeID'],[StandardScaler(),PolynomialFeatures(2)]),
             
            (['age','userID'],[MinMaxScaler(),StandardScaler(),PolynomialFeatures(2)])
            ],None)
         
tests=mapper.fit_transform(tests)
 '''
#设置参数
parms = {
    'booster': 'gbtree',
    # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
    'objective': 'binary:logistic',
    'gamma': 0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'max_depth': 8,  # 构建树的深度 [1:] 最佳
    # 'lambda':450,  # L2 正则项权重
   # 'subsample': 0.8,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_bytree': 0.7,  # 构建树树时的采样比率 (0:1]
   # 'min_child_weight':8, # 节点的最少特征数
    'lambda ':0.5,
    'silent': 1,
    'eta': 0.04,  # 如同学习率
    'seed': 710,
    'nthread': 4,
    'min_child_weight':6, #最佳
    'eval_metric':'logloss'
}

#
plst = list(parms.items()) 
#offerset = 10000
#endset = 3040000
#训练数据划分8：1：1 训练集：验证集：测试集
xgTrain = xgb.DMatrix(trains[:],label=labels[:])
#xgval = xgb.DMatrix(trains[offerset:],label=labels[offerset:])
#xgTest = xgb.DMatrix(trains[endset:],label=labels[endset:])
#真正的测试集
tests = pd.DataFrame(tests)
xgTest = xgb.DMatrix(tests)
#验证集错误率
watchList = [(xgTrain,'train')]

#设置迭代次数
num_iter = 210
#训练模型

model = xgb.train(plst,xgTrain,num_boost_round=num_iter,evals=watchList,early_stopping_rounds=150,maximize=False)
#model.save_model(r'E:\tencent\result\model_0.0943')
#预测
#pre = model.predict(xgTest)
#pre = model.predict(xgTest,ntree_limit=model.best_iteration)

#pre = pd.DataFrame(pre,columns=['prod']) 
#pre.to_csv(r'E:\tencent\result\result1.csv')

cost_time = time.time() - now
print("end ......", '\n', "cost time:", cost_time, "(s)......")

                      
 

