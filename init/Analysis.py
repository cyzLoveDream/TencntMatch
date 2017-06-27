# -*- coding: utf-8 -*-
"""
Created on Sat May 13 09:34:43 2017

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

df_train = pd.read_pickle(r'E:\tencent\input\noonehot\train')
df_test = pd.read_pickle(r'E:\tencent\input\noonehot\test') #
#处理时间特征
df_train['day']=df_train['clickTime'].map(lambda x:  int(x/10000))
df_train['hour'] = (df_train['clickTime']-df_train['day']*10000).map(lambda x: int( x/100))
df_train['minute'] =(df_train['clickTime']-(df_train['day']*10000+df_train['hour']*100)).map(lambda x: int (x/10))
df_train['day_hour'] = (df_train['day']*10000 + df_train['hour']*100).map(lambda x: int(x))
df_train['hour_minute'] = df_train['hour']*10 + df_train['minute']
df_train['pre_day'] = df_train['day']-1
df_train['age_split'] = df_train['age'].map(lambda x: int(x / 5))
#处理测试集
df_test['day']=df_test['clickTime'].map(lambda x:  int(x/10000))
df_test['hour'] = (df_test['clickTime']-df_test['day']*10000).map(lambda x: int( x/100))
df_test['minute'] =(df_test['clickTime']-(df_test['day']*10000+df_test['hour']*100)).map(lambda x: int (x/10))
df_test['day_hour'] = (df_test['day']*10000 + df_test['hour']*100).map(lambda x: int(x))
df_test['hour_minute'] = df_test['hour']*10 + df_test['minute']
df_test['pre_day'] = df_test['day']-1
df_test['age_split'] = df_test['age'].map(lambda x: int(x / 5))
def cntDualKey(df, vn, vn2, key_src, key_tgt, fill_na=False):
    
    print("build src key")
    _key_src = np.add(df[key_src].astype('str').values, df[vn].astype('str').values)
    print ("build tgt key")
    _key_tgt = np.add(df[key_tgt].astype('str').values, df[vn].astype('str').values)
    
    if vn2 is not None:
        _key_src = np.add(_key_src, df[vn2].astype('str').values)
        _key_tgt = np.add(_key_tgt, df[vn2].astype('str').values)

    print( "aggreate by src key")
    grp1 = df.groupby(_key_src)
    cnt1 = grp1[vn].aggregate(np.size)
    
    print ("map to tgt key")
    vn_sum = 'sum_' + vn + '_' + key_src + '_' + key_tgt
    _cnt = cnt1[_key_tgt].values

    if fill_na is not None:
        print ("fill in na")
        _cnt[np.isnan(_cnt)] = fill_na    

    vn_cnt_tgt = 'cnt_' + vn + '_' + key_tgt
    if vn2 is not None:
        vn_cnt_tgt += '_' + vn2
    df[vn_cnt_tgt] = _cnt
      
def my_grp_cnt(group_by, count_by):
    _ts = time.time()
    _ord = np.lexsort((count_by, group_by))
    print(time.time() - _ts)
    _ts = time.time()    
    _ones = pd.Series(np.ones(group_by.size))
    print(time.time() - _ts)
    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    runnting_cnt = 0
    for i in range(1, group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            if count_by[_ord[i-1]] != count_by[i0]: 
                running_cnt += 1
        else:
            running_cnt = 1
            _prev_grp = group_by[i0]
        if i == group_by.size - 1 or group_by[i0] != group_by[_ord[i+1]]:
            j = i
            while True:
                j0 = _ord[j]
                _cs1[j0] = running_cnt
                if j == 0 or group_by[_ord[j-1]] != group_by[j0]:
                    break
                j -= 1
            
    print(time.time() - _ts)
    if True:
        return _cs1
    else:
        _ts = time.time()    

        org_idx = np.zeros(group_by.size, dtype=np.int)
        print(time.time() - _ts)
        _ts = time.time()    
        org_idx[_ord] = np.asarray(range(group_by.size))
        print(time.time() -_ts)
        _ts = time.time()    

        return _cs1[org_idx]
 

cntDualKey(df_train,'userID',None,'day_hour','day_hour',fill_na=0)
df_train['camgaignID_cnt_by_advertiserID'] = my_grp_cnt(df_train.camgaignID.values.astype('str'),df_train.advertiserID.values.astype('str'))
df_train['adID_cnt_by_camgaignID'] = my_grp_cnt(df_train.camgaignID.values.astype('str'),df_train.adID.values.astype('str'))
df_train['creativeID_cnt_by_adID'] = my_grp_cnt(df_train.creativeID.values.astype('str'),df_train.adID.values.astype('str'))
df_train['camgaignID_cnt_by_appID'] = my_grp_cnt(df_train.camgaignID.values.astype('str'),df_train.appID.values.astype('str'))

column = ['creativeID','appID','camgaignID','positionID','haveBaby','education','advertiserID','hometown','adID'
          ,'503','201','106','402','203','209','407','409','marriageStatus','104','108','408','406'
        ,'403','211','0_x']
tm = ['day','hour','minute']
for col in column:
    for t in tm:
        c = col + "_"+"_by_" + t
        df_train[c] = my_grp_cnt(df_train[col].values.astype('str'),df_train[t].values.astype('str'))
        print(t+" and "+col + "done" )

df_train.to_pickle(r'E:\tencent\input\lr\train')
#测试集

cntDualKey(df_test,'userID',None,'day_hour','day_hour',fill_na=0)
df_test['camgaignID_cnt_by_advertiserID'] = my_grp_cnt(df_test.camgaignID.values.astype('str'),df_test.advertiserID.values.astype('str'))
df_test['adID_cnt_by_camgaignID'] = my_grp_cnt(df_test.camgaignID.values.astype('str'),df_test.adID.values.astype('str'))
df_test['creativeID_cnt_by_adID'] = my_grp_cnt(df_test.creativeID.values.astype('str'),df_test.adID.values.astype('str'))
df_test['camgaignID_cnt_by_appID'] = my_grp_cnt(df_test.camgaignID.values.astype('str'),df_test.appID.values.astype('str'))
for col in column:
    for t in tm:
        c = col + "_"+"_by_" + t
        df_test[c] = my_grp_cnt(df_test[col].values.astype('str'),df_test[t].values.astype('str'))


df_test.to_pickle(r'E:\tencent\input\lr\test')
