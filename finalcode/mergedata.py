# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 22:02:14 2017

@author: Administrator
"""

import pandas as pd
import time
import gc

for i in range(17,32):
    now = time.time()
    print("begin merge data for day {0}".format(i))
    now1=now
    paths = "E:\\finaldata\\final\\splitdata\\train_" + str(i)
    data = pd.read_pickle(paths)
    user = pd.read_csv(r'E:\finaldata\final\user.csv')
    print("begin merge user")
    data = data.merge(user,how ='left',on='userID')
    del user
    print("user finish merge in {0}".format(time.time()-now))
    now = time.time()
    ad = pd.read_csv(r'E:\finaldata\final\ad.csv')
    print("begin merge ad")
    data = data.merge(ad,how='left',on='creativeID')
    del ad
    print("ad finish merge in {0}".format(time.time()-now))
    now = time.time()
    gc.collect()
    position = pd.read_csv(r'E:\finaldata\final\position.csv')
    print("begin merge position")
    data = data.merge(position,how='left',on='positionID')
    del position
    print("position merge finish in {0}".format(time.time()-now))
    now = time.time()
    appCate = pd.read_csv(r'E:\finaldata\final\app_categories.csv')
    print("begin merge appCate")
    data = data.merge(appCate,how ='left',on='appID')
    del appCate
    gc.collect()
    print("finish merge appCate in {0}".format(time.time()-now))
    data.to_pickle(paths)
    print("finish merge data for day {0} ,the data shape in {1},the time is {2}".format(i,data.shape[0],time.time()-now1))
    del data
    gc.collect()
print("begin fea raw train data")
train = pd.DataFrame()
for i in range(27,31):
    print("begin handle data the day",i)
    paths = "E:\\finaldata\\final\\splitdata\\train_" +str(i)
    data = pd.read_pickle(paths)
    train = pd.concat([train,data])
    del data
    gc.collect()
train.to_pickle(r'E:\finaldata\final\sample\rawdata\train')
print("finish raw data size {0} * {1}".format(train.shape[0],train.shape[1]))
del train
gc.collect()
print("begin fea raw test data")
test = pd.read_pickle(r'E:\finaldata\final\splitdata\train_31')
test.to_pickle(r'E:\finaldata\final\sample\rawdata\test')
print("finish raw test data size {0} * {1}".format(test.shape[0],test.shape[1]))
del test
gc.collect()
    