# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:50:02 2017

@author: Administrator
"""
#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import time
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
#%matplotlib inline
import os
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
#测试方法
now = time.time() 
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        print('beginCV')
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='logloss', early_stopping_rounds=early_stopping_rounds, show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0]) 
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['label'],eval_metric='logloss') 
    #预测 
    dtrain_predictions = alg.predict_proba(dtrain[predictors])[:,1]
    #Print model report:
    print(time.time()-now)
    print ("\nModel Report")
    print("logloss (Train): %f",metrics.log_loss(dtrain['label'],dtrain_predictions))
    '''             
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    '''
#读取数据

train_data = pd.read_pickle(r'E:\tencent\input\noonehot\train')
print(1)
#test_data = pd.read_pickle(r'E:\tencent\input\noonehot\test')
print(2) 
#先分类
train_data['appCategory_1'] = train_data['appCategory'].map(lambda x: round(x / 100))   # 列处理
train_data['appCategory_2'] = train_data['appCategory'].map(lambda x: x % 100)
train_data['home_province'] = train_data['hometown'].map(lambda x: round(x / 100))
train_data['home_city'] = train_data['hometown'].map(lambda x: x % 100)
train_data['live_province'] = train_data['residence'].map(lambda x: round(x / 100))
train_data['live_city'] = train_data['residence'].map(lambda x: x % 100)
train_data=train_data[(train_data['clickTime']>270000) & (train_data['clickTime']<290000)]
#合特征文件1
i=0
for file in os.listdir(r'E:\tencent\trainfeaturecount'):
    i+=1
    filename = "E:\\tencent\\trainfeaturecount\\"+file
    index = file.index('.')
    f = file[:index]
    data = pd.read_csv(filename)
    train_data = train_data.merge(data,how='left',on=f) 
    print(i)  
train_data = train_data.fillna(value=0) 
train_data.to_pickle(r'E:\tencent\input\chouyang\train2728')
#获取训练集L 
train_data = train_data.drop('conversionTime',axis=1)
#trains = train_data.drop('f1')
#labels = train_data.iloc[:,:1].values
#trains = train_data.iloc[:,1:]
target='label'
con = 'conversionTime'
predictors = [x for x in train_data.columns if x not in [target, con]]
print('begin调参')
#X_train, X_test, y_train, y_test = train_test_split(trains,labels,test_size = 0.2,random_state = 42)
gama = [0.0,0.1,0.2,0.3,0.4,0.5]
    for g in gama: 
    print('gama:',g)
    xgb1 = XGBClassifier(
            learning_rate =0.1,
            n_estimators=1000,
            max_depth=14,
            min_child_weight=3,
            gamma=g,
            subsample=0.8,
            colsample_bytree=0.8,
            objective= 'binary:logistic',
            nthread=7,
            scale_pos_weight=1,
            seed=780)
    modelfit(xgb1, train_data, predictors)   

