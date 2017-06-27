# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import random
import pandas as pd
import xgboost as xgb
import numpy as np
import time 
import sklearn_pandas as sp 
from sklearn_pandas import DataFrameMapper
# transforms for category variables
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder

# transformer for numerical variables
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

# transformer for combined variables
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

# user-defined transformers
from sklearn.preprocessing import FunctionTransformer  

now = time.time() 
#trainData = train_data[['label','gender','appID']]

# 添加交叉特征
#train_data.to_csv()
# 添加特征
'''
"label","clickTime","conversionTime","creativeID","userID","positionID",
"connectionType","telecomsOperator","age","gender","education","marriageStatus","haveBaby",
"hometown","residence","advertiserID","appID","appPlatform","sitesetID","positionType"
'''
train_data = pd.read_csv(r'E:\tencent\localTest\localtrain.csv')
#test_data = pd.read_csv(r'E:\tencent\pre\testresult.csv')
#处理空值
#读取数据 
train_data = train_data.fillna(0)
#开始处理特征
featureData = train_data.iloc[:,3:]
mapper = DataFrameMapper([
            ('education',LabelBinarizer()),
            (['positionID','connectionType','telecomsOperator','gender','education','marriageStatus',
              'haveBaby','appPlatform','sitesetID','positionType'],OneHotEncoder()),
            (['hometown','residence'],[FunctionTransformer(lambda x: x%100),MultiLabelBinarizer()]), #未解决 
            (['creativeID'],[StandardScaler(),PolynomialFeatures(2)]), 
            (['age','userID'],[MinMaxScaler(),StandardScaler(),PolynomialFeatures(2)])
            ],None)
featureData = mapper.fit_transform(featureData)

#获取训练集L
#trains = train_data.drop('f1')
labels = train_data.iloc[:,:1].values
trains = featureData
#trains = trains.drop('conversionTime',axis=1).values
                        
#获取测试集
#tests = test_data.iloc[:,3:]
#设置参数
parms = {
    'booster': 'gbtree',
    # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
    'objective': 'binary:logistic',
    'gamma': 0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'max_depth': 8,  # 构建树的深度 [1:] 最佳
    # 'lambda':450,  # L2 正则项权重
    'subsample': 0.8,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
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
offerset = 80
#endset = 3040000
#训练数据划分8：1：1 训练集：验证集：测试集
xgTrain = xgb.DMatrix(trains[:offerset],label=labels[:offerset])
xgval = xgb.DMatrix(trains[offerset:],label=labels[offerset:])
#xgTest = xgb.DMatrix(trains[endset:],label=labels[endset:])
#真正的测试集
#tests = pd.DataFrame(tests)
#xgTest = xgb.DMatrix(tests)
#验证集错误率
watchList = [(xgTrain,'train'),(xgval,'val')]

#设置迭代次数
num_iter = 10
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

                      
 