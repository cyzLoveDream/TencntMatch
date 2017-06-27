# -*- coding: utf-8 -*-
"""
Created on Sat May 13 12:29:05 2017

@author: Administrator
"""
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn_pandas import DataFrameMapper
#对测试集做同样的变化
now1 = time.time()
testData = pd.read_csv(r'E:\tencent\input\testresult.csv')
print('读取文件时间：',time.time()-now1)
now = time.time()
cot = pd.get_dummies(testData.connectionType)
telo = pd.get_dummies(testData.telecomsOperator)
gend = pd.get_dummies(testData.gender)
edu = pd.get_dummies(testData.education)
ms = pd.get_dummies(testData.marriageStatus)
hb = pd.get_dummies(testData.haveBaby)
aid = pd.get_dummies(testData.advertiserID)
apf = pd.get_dummies(testData.appPlatform)
sid = pd.get_dummies(testData.sitesetID)
pt = pd.get_dummies(testData.positionType)
cid = pd.get_dummies(testData.camgaignID)
ac = pd.get_dummies(testData.appCategory)
print('testonehottime:' , time.time()-now)
now =time.time()
#丢弃原有的列
testData = testData.drop('connectionType',axis=1)
testData = testData.drop('telecomsOperator',axis=1)
testData = testData.drop('gender',axis=1)
testData = testData.drop('education',axis=1)
testData = testData.drop('marriageStatus',axis=1)
testData = testData.drop('haveBaby',axis=1)
testData = testData.drop('advertiserID',axis=1)
testData = testData.drop('appPlatform',axis=1)
testData = testData.drop('sitesetID',axis=1)
testData = testData.drop('positionType',axis=1)
testData = testData.drop('camgaignID',axis=1)
testData = testData.drop('appCategory',axis=1)
print('testdroptime:' , time.time()-now)
now =time.time()
#合并特征
testData = pd.concat([testData,cot,telo,gend,edu,ms,hb,aid,apf,sid,pt,cid,ac],axis=1)
print('testmergetime:' , time.time()-now)
now =time.time()
#进行归一化 age
'''
mm = MinMaxScaler()
age = mm.fit(testData.age)
age = pd.DataFrame(age)
testData = testData.drop('age',axis=1)
testData = pd.concat(testData,age)
print('testagetime:' , time.time()-now)

now =time.time()
'''
testData.to_csv(r'E:\tencent\output\test.csv',index = False)
print('alltime:' , time.time()-now1)
del cot
del edu
del ms
del hb
del aid
del apf
del sid
del pt
del cid
del ac