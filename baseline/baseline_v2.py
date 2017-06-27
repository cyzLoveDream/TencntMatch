# -*- coding: utf-8 -*-
"""
baseline 2: ad.csv (creativeID/adID/camgaignID/advertiserID/appID/appPlatform) + lr
"""

import zipfile
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import cross_validation
import time
import os

now = time.time()
# load data
data_root = "."
'''
alldata = pd.read_pickle(r'E:\tencent\input\nofeature\alldatafeaturenext')
alldata.drop('clickTime',axis=1,inplace = True)
dfTrain = alldata[alldata['day']!=31]
dfTest = alldata[alldata['day']==31]
'''

dfTrain = pd.read_pickle(r'E:\tencent\input\lr\train')
dfTest = pd.read_pickle(r'E:\tencent\input\lr\test')
#dfTrain.drop('conversionTime',axis=1,inplace=True)
dfTrain.drop('clickTime',axis=1,inplace=True)
dfTest.drop('label',axis =1,inplace=True)
dfTest['instanceID'] = range(1,dfTest.shape[0]+1)
#dfTest.drop('clickTime',axis=1,inplace=True)
print('导入数据完成')
#dfTrain = dfTrain.sample(frac=0.2)
dfTrain = dfTrain.fillna(value=0)
dfTest = dfTest.fillna(value=0)
# process data
y_train = dfTrain["label"].values
 
# feature engineering/encoding
#合文件特征
'''
#合特征文件1 
i=0
for file in os.listdir(r'E:\tencent\trainfeaturedoublecount'):
    i+=1
    fn = "E:\\tencent\\trainfeaturedoublecount\\"+file
    index = file.index('.')
    f = file[:index]
    data = pd.read_csv(fn)
    data.fillna(value=0,inplace = True)
    dfTrain = dfTrain.merge(data,how='left',on=f)
    dfTest = dfTest.merge(data,how='left',on=f)
    print(i)
print('合并文件1完成') 
'''
#合特征文件2 
'''
i=0
for file in os.listdir(r'E:\tencent\1'):
    i+=1
    fn = "E:\\tencent\\1\\"+file
    index = file.index('.')
    f = file[:index]
    data = pd.read_csv(fn)
    dfTrain = dfTrain.merge(data,how='left',on=f)
    dfTest = dfTest.merge(data,how='left',on=f)
    print(i)
print('合并文件2完成')

#合并文件3
i=0
for file in os.listdir(r'E:\tencent\trainFeature'):
    i+=1
    fn = "E:\\tencent\\trainFeature\\"+file
    index = file.index('.')
    f = file[:index]
    data = pd.read_csv(fn)
    data.fillna(value=0,inplace = True)
    dfTrain = dfTrain.merge(data,how='left',on=f)
    dfTest = dfTest.merge(data,how='left',on=f)
    print(i)
print('合并文件3完成')
'''
#自身特征分裂
#处理训练集

dfTrain['appCategory_app'] = dfTrain['appCategory'].map(lambda x: round(x / 100))   # 列处理
dfTrain['appCategory_appc'] = dfTrain['appCategory'].map(lambda x: x % 100)
dfTrain['home_province'] = dfTrain['hometown'].map(lambda x: round(x / 100))
dfTrain['home_city'] = dfTrain['hometown'].map(lambda x: x % 100)
dfTrain['live_province'] = dfTrain['residence'].map(lambda x: round(x / 100))
dfTrain['live_city'] = dfTrain['residence'].map(lambda x: x % 100)
dfTrain['age_sex'] = dfTrain['age'] + dfTrain['gender']*100
dfTrain.drop(['appCategory','hometown','residence'],inplace = True,axis =1)
#处理测试集
dfTest['appCategory_app'] = dfTest['appCategory'].map(lambda x: round(x / 100))   # 列处理
dfTest['appCategory_appc'] = dfTest['appCategory'].map(lambda x: x % 100)
dfTest['home_province'] = dfTest['hometown'].map(lambda x: round(x / 100))
dfTest['home_city'] = dfTest['hometown'].map(lambda x: x % 100)
dfTest['live_province'] = dfTest['residence'].map(lambda x: round(x / 100))
dfTest['live_city'] = dfTest['residence'].map(lambda x: x % 100)
dfTest['age_sex'] = dfTest['age'] + dfTest['gender']*100
dfTest.drop(['appCategory','hometown','residence'],inplace = True,axis =1)
print('列处理完成')

'''
#合并文件4
i=0
for file in os.listdir(r'E:\tencent\trainfeaturecount'):
    i+=1
    fn = "E:\\tencent\\trainfeaturecount\\"+file
    index = file.index('.')
    f = file[:index]
    data = pd.read_csv(fn)
    dfTrain = dfTrain.merge(data,how='left',on=f)
    dfTest = dfTest.merge(data,how='left',on=f)
    print(i)
print('合并文件4完成')

'''

#onehot
enc = OneHotEncoder()
feats = ['creativeID', 'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform'
         ,'connectionType','telecomsOperator','age','gender','education','marriageStatus','haveBaby'
         ,'advertiserID','sitesetID','positionType','appCategory_app'
         ,'appCategory_appc','home_province','home_city','live_province','live_city']
for i,feat in enumerate(feats):
    print(feat)
    x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1, 1))
    x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
    if i == 0:
        X_train, X_test = x_train, x_test
    else:
        X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))
print('finish onehot')
'''

#随机划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size = 0.3,random_state = 42)
# model training

lr = LogisticRegression()
#score = cross_validation.cross_val_score(lr,X_train,y_train,cv=10,scoring='neg_log_loss')
#print(score)
'''
lr = LogisticRegression()
lr.fit(X_train, y_train)
proba_test = lr.predict_proba(X_test)[:,1] 
#print("logloss (Train): %f",metrics.log_loss(y_test,proba_test))
print('proba finish')
#print("logloss (Train): %f",metrics.log_loss(y_test,proba_test))


# submission

df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv("submission.csv", index=False)
with zipfile.ZipFile("submission.zip", "w") as fout:
    fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)

print("总时间",time.time()-now)
