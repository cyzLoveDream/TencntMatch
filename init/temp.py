# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
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
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
import os
import gc
import base_cnt_fea



now = time.time() 
'''
data_2 = pd.read_pickle(r'E:\tencentfinal\final\final\splitbyday\train_24')
data_1 = pd.read_pickle(r'E:\tencentfinal\final\final\splitbyday\train_25')
data_0 = pd.read_pickle(r'E:\tencentfinal\final\final\splitbyday\train_26')
data0 = pd.read_pickle(r'E:\tencentfinal\final\final\splitbyday\train_27')
data1 = pd.read_pickle(r'E:\tencentfinal\final\final\splitbyday\train_28')
data2 = pd.read_pickle(r'E:\tencentfinal\final\final\splitbyday\train_29')
#val_data = pd.read_pickle(r'E:\tencentfinal\final\final\splitbyday\train_30')
test_data = pd.read_pickle(r'E:\tencentfinal\final\final\splitbyday\train_31')
train_data = pd.concat([data_2,data_1,data_0,data0,data1,data2]) 
train_data = train_data.sample(frac=0.4)
del data0
del data_0
del data_1
del data_2
del data1
del data2 
gc.collect()
'''
data = pd.read_csv(r'E:\finaldata\final\sample\feadata\data.csv')
train_data =data[data['day']<=30]
test_data = data[data['day']==31]
#val_data = alldata[alldata['day'] == 29]
del data
gc.collect()
print(1) 
#处理空值
test_data = test_data.fillna(value=0)
train_data = train_data.fillna(value=0) 
#train_data.drop('conversionTime',axis=1,inplace=True)
#test_data = test_data.drop('instanceID',axis =1,inplace = True)
#train_data = pd.concat([train_data,test_data4])
print("data input finish in size")
#获取训练集L
'''
i=0
for file in os.listdir(r'E:\tencent\trainfeaturedoublecount'):
    i+=1
    fn = "E:\\tencent\\trainfeaturedoublecount\\"+file
    index = file.index('.')
    f = file[:index]
    data = pd.read_csv(fn)
    data.fillna(value=0,inplace = True)
    train_data = train_data.merge(data,how='left',on=f)
    test_data = test_data.merge(data,how='left',on=f)
    print(i)
print('合并文件1完成') 
'''
'''
#trains = train_data.drop('f1')
train_data['appCategory_app'] = train_data['appCategory'].map(lambda x: round(x / 100))   # 列处理
train_data['appCategory_appc'] = train_data['appCategory'].map(lambda x: x % 100)
train_data['home_province'] = train_data['hometown'].map(lambda x: round(x / 100))
train_data['home_city'] = train_data['hometown'].map(lambda x: x % 100)
train_data['live_province'] = train_data['residence'].map(lambda x: round(x / 100))
train_data['live_city'] = train_data['residence'].map(lambda x: x % 100)
train_data['age_sex'] = train_data['age'] + train_data['gender']*100
#时间特征
print("begin train time feature")
train_data['hour'] = (train_data['clickTime']-train_data['day']*1000000).map(lambda x: int( x/10000))
train_data['minute'] =(train_data['clickTime']-(train_data['day']*1000000+train_data['hour']*10000)).map(lambda x: int (x/1000))
train_data['day_hour'] = (train_data['day']*1000000 + train_data['hour']*10000).map(lambda x: int(x))
train_data['hour_minute'] = train_data['hour']*1000 + train_data['minute']
train_data['pre_day'] = train_data['day']-1
train_data['age_split'] = train_data['age'].map(lambda x: int(x / 5))
print("begin train time feature")
print("drop feature")
train_data.drop(['userID','pre_day'],axis = 1,inplace=True)
'''
#处理测试集
'''
test_data = train_data[train_data['day']==31]
train_data = train_data[train_data['day']!=31]
'''

'''
test_data['appCategory_app'] = test_data['appCategory'].map(lambda x: round(x / 100))   # 列处理
test_data['appCategory_appc'] = test_data['appCategory'].map(lambda x: x % 100)
test_data['home_province'] = test_data['hometown'].map(lambda x: round(x / 100))
test_data['home_city'] = test_data['hometown'].map(lambda x: x % 100)
test_data['live_province'] = test_data['residence'].map(lambda x: round(x / 100))
test_data['live_city'] = test_data['residence'].map(lambda x: x % 100)
test_data['age_sex'] = test_data['age'] + test_data['gender']*100
print("begin test time feature")
test_data['hour'] = (test_data['clickTime']-test_data['day']*1000000).map(lambda x: int( x/10000))
test_data['minute'] =(test_data['clickTime']-(test_data['day']*1000000+train_data['hour']*10000)).map(lambda x: int (x/1000))
test_data['day_hour'] = (test_data['day']*1000000 + test_data['hour']*10000).map(lambda x: int(x))
test_data['hour_minute'] = test_data['hour']*1000 + test_data['minute']
test_data['pre_day'] = test_data['day']-1
test_data['age_split'] = test_data['age'].map(lambda x: int(x / 5))
'''
'''
features = ['camgaignID','residence','creativeID','education','age']
for f in features:
  #  f_pd  = base_cnt_fea.get_all_cnt_fea(f)
    path = "E:\\tencentfinal\\final\\cntfeature\\" + f +".txt"
    f_pd = pd.read_csv(path)
    train_data = train_data.merge(f_pd,how='left',on=f)
  #  train_data.drop(f,axis=1,inplace=True)
    test_data = test_data.merge(f_pd,how='left',on=f)
  #  test_data.drop(f,axis=1,inplace=True)
    del f_pd
    gc.collect()
'''
'''
#train data
train_data['hour'] = (train_data['clickTime']-train_data['day']*1000000).map(lambda x: int( x/10000))
train_data['minute'] =(train_data['clickTime']-(train_data['day']*1000000+train_data['hour']*10000)).map(lambda x: int (x/1000))
train_data['day_hour'] = (train_data['day']*1000000 + train_data['hour']*10000).map(lambda x: int(x))
train_data['hour_minute'] = train_data['hour']*1000 + train_data['minute']
train_data['home_province'] = train_data['hometown'].map(lambda x: round(x / 100))
train_data['home_city'] = train_data['hometown'].map(lambda x: x % 100)
train_data['age_sex'] = train_data['age'] + train_data['gender']*100
train_data['age_split'] = train_data['age'].map(lambda x: int(x / 5))
#test data
test_data['hour'] = (test_data['clickTime']-test_data['day']*1000000).map(lambda x: int( x/10000))
test_data['minute'] =(test_data['clickTime']-(test_data['day']*1000000+test_data['hour']*10000)).map(lambda x: int (x/1000))
test_data['day_hour'] = (test_data['day']*1000000 + test_data['hour']*10000).map(lambda x: int(x))
test_data['hour_minute'] = test_data['hour']*1000 + test_data['minute']
test_data['home_province'] = test_data['hometown'].map(lambda x: round(x / 100))
test_data['home_city'] = test_data['hometown'].map(lambda x: x % 100)
test_data['age_sex'] = test_data['age'] + test_data['gender']*100
test_data['age_split'] = test_data['age'].map(lambda x: int(x / 5))
#train data
train_data['appCategory_app'] = train_data['appCategory'].map(lambda x: round(x / 100))   # 列处理
train_data['appCategory_appc'] = train_data['appCategory'].map(lambda x: x % 100)
test_data['appCategory_app'] = test_data['appCategory'].map(lambda x: round(x / 100))   # 列处理
test_data['appCategory_appc'] = test_data['appCategory'].map(lambda x: x % 100)
train_data.drop(['appCategory','hometown'],axis=1,inplace=True)
test_data.drop(['appCategory','hometown'],axis=1,inplace=True)
train_data['residence_province'] = train_data['residence'].map(lambda x: round(x / 100))
train_data['residence_city'] = train_data['residence'].map(lambda x: x % 100)
test_data['residence_province'] = test_data['residence'].map(lambda x: round(x / 100))
test_data['residence_city'] = test_data['residence'].map(lambda x: x % 100)
test_data.drop(['residence','age'],axis=1,inplace=True)
train_data.drop(['residence','age'],axis=1,inplace=True)

print("finish product train feature {0},{1}".format(train_data.shape[0],train_data.shape[1]))
print("finish product test feature {0},{1}".format(test_data.shape[0],test_data.shape[1]))
print('列处理完成')
'''
#合并文件4
'''
i=0
for file in os.listdir(r'E:\tencent\trainfeaturecount'):
    i+=1
    fn = "E:\\tencent\\trainfeaturecount\\"+file
    index = file.index('.')
    f = file[:index]
    data = pd.read_csv(fn)
    data = data.fillna(value=0)
    train_data = train_data.merge(data,how='left',on=f)
    test_data = test_data.merge(data,how='left',on=f)
    print(i)
print('合并文件4完成')
'''
features = [x for x in train_data.columns if x not in ['label','instanceID','day','clickTime','userID','conversionTime','_bindApp_fea']]
#trains = trains.drop('conversionTime',axis=1).values 
#trains = train_data[features]
#labels = train_data['label'].values
val_data = train_data[train_data['day']==30]
train_data = train_data[train_data['day']!=30] 
y_train = train_data['label'].values 
X_train = train_data[features]
X_val = val_data[features]
y_val = val_data['label'].values
del train_data
del val_data
 
#X_train, X_val, y_train, y_val = train_test_split(trains,labels,test_size = 0.1,random_state = 42) 
# 0.02  0.110 
gc.collect()
#获取测试集

print(4)
#设置参数
parms = {
    'booster': 'gbtree',
    # 这里二分类，是一个二类的问题，因此采用了binary:logistic分类器，
    'objective': 'binary:logistic',
    'gamma': 0.2,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'scale_pos_weight':1,
    'lambda':2,
    'max_depth': 10,  # 构建树的深度 [1:] 最佳
    # 'lambda':450,  # L2 正则项权重
    'subsample': 1,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_bytree': 0.7,  # 构建树树时的采样比率 (0:1]
   # 'min_child_weight':8, # 节点的最少特征数 
    'silent': 1,
    'eta': 0.1,  # 如同学习率
    'seed': 780,
    'nthread': 7,
    'max_delta_step':6,
    'min_child_weight': 3, #最佳
    'eval_metric':'logloss',
    'n_estimators':200
} 

#
plst = list(parms.items()) 
#endset = 1300000
#训练数据划分8：1：1 训练集：验证集：测试集
#y_train = train_data['label'].values
#X_train = train_data.iloc[:,2:]
xgTrain = xgb.DMatrix(X_train,label=y_train)
del X_train
del y_train
gc.collect()
print(5)
#y_val = val_data['label'].values
#X_val = val_data.iloc[:,2:]
xgval = xgb.DMatrix(X_val,label=y_val)
del X_val
del y_val
gc.collect()
print(6)
#xgTest = xgb.DMatrix(trains[endset:],label=labels[endset:])
#真正的测试集
#tests = pd.DataFrame(tests)
tests = test_data[features]
del test_data
xgTest = xgb.DMatrix(tests)

gc.collect()
print(7)
#验证集错误率
watchList = [(xgTrain,'train'),(xgval,'val')] 
#设置迭代次数
num_iter = 10000
#训练模型

model = xgb.train(plst,xgTrain,num_boost_round=num_iter,evals=watchList,early_stopping_rounds=50,maximize=False)  
ptrain = model.predict(xgTrain, output_margin=True)
ptest  = model.predict(xgTest, output_margin=True)
xgTrain.set_base_margin(ptrain)
xgTest.set_base_margin(ptest)

model1 = xgb.train(plst,xgTrain,num_boost_round=num_iter,evals=watchList,early_stopping_rounds=50,maximize=False)  
model1.save_model(r'E:\finaldata\model\modeltrain')
print ('this is result of running from initial prediction')
#预测
#pre = model.predict(xgTest)
pre = model1.predict(xgTest,ntree_limit=model.best_iteration)
print("finish pred mean :{0}".format(pre.mean()))
del xgTest
gc.collect() 
del tests
gc.collect()  
#xgb.plot_importance(model)
p=pd.DataFrame({"increaseid":range(1,1+len(pre)),"prob":pre}) 
p.to_csv(r'E:\finaldata\result\submission.csv',index=False)

cost_time = time.time() - now
print("end ......", '\n', "cost time:", cost_time, "(s)......")

                      
 
