# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 21:08:13 2017

@author: adc
"""

import pandas as pd
import time
import gc

now = time.time()
now1 = now 
data = pd.read_csv(r'E:\finaldata\final\train.csv')
print("data size is {0}*{1}".format(data.shape[0],data.shape[1]))
data['day'] = data['clickTime'].map(lambda x: round(x/1000000)) 
for i in range(17,32):
    print("begin handle data the day",i)
    paths = "E:\\finaldata\\final\\splitdata\\train_" +str(i)
    train_data = data[data['day']==i]
    train_data.to_pickle(paths) 
    print("finish handle data the day {0} in time {1},the data size is {2}*{3}".format(i,time.time()-now,train_data.shape[0],train_data.shape[1]))
    del train_data
    gc.collect()
    now = time.time()
print("finish splitdata in time {0}".format(time.time()-now1))
test = pd.read_csv(r'E:\finaldata\final\test.csv')
test.to_pickle(r'E:\finaldata\final\splitdata\train_31')