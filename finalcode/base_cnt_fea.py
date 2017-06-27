# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 14:49:27 2017

@author: Administrator
"""

import pandas as pd
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
import time
import gc

def get_all_cnt_fea(feature,begin =17,end =31):
    mms = MinMaxScaler()
    now = time.time()
    fea_cnt = pd.DataFrame()
    for i in range(begin,end):
        print("begin read data")
        paths = "E:\\tencentfinal\\final\\final\\splitbyday\\train_"+str(i)
        data = pd.read_pickle(paths)
        print("data read finished, the data size is {0} and {1}".format(data.shape[0],data.shape[1]))
        fea_dic = {}
        print("begin handle day {0}".format(i))
        a = 1
        for line in range(0,data.shape[0]): 
            if a % 500000 ==0 :
               print("have been handle data size :",a)
            a += 1
            data_in_pd = data.at[line,feature]
            if data_in_pd not in fea_dic:
                fea_dic.setdefault(data_in_pd,1)
            else:
                fea_dic[data_in_pd] += 1 
        print("data finish handle in size:",a)
        col = feature + "_cnt"
        fea_pd = pd.DataFrame(Series(fea_dic),columns=[col])
        del fea_dic
        del col
        if i==begin:
            fea_cnt = pd.concat([fea_cnt,fea_pd])
        fea_cnt.add(fea_pd,fill_value=0)
        del fea_pd
        print("the day {0} have been handled in time {1}".format(i,time.time()-now))
        del data
        del a
        now = time.time()
        gc.collect()
    print("the feature has been counted from day {0} to day {1} in time {2}".format(begin,end,time.time()-now))
    print("Min max scaler") 
    ix = fea_cnt.index
    col = fea_cnt.columns
    fea_cnt = pd.DataFrame(mms.fit_transform(fea_cnt),index =ix,columns = col ) 
    return fea_cnt
features = 'camgaignID'
f_pd  = get_all_cnt_fea(features)

            
        
    