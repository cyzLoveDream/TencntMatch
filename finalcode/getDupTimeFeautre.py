# -*- coding: utf-8 -*-
"""
Created on Biorad 25 07:25:49 2017

@author: Biorad
"""

import pandas as pd
import numpy as np
import time
import math
import gc

def get_dupTime_fea(df,user = 'userID',creative = 'creativeID',click = 'clickTime',print_size = 1000000):
    _time = time.time()
    _totalTime = _time
    data = df[[user,creative,click]].sort_values(by=[user,creative,click])
    ur = data[user]
    ce = data[creative]
    ck = data[click]
    ix = list(data.index)
    _sub_with_first = pd.Series(np.zeros(data.shape[0]),index = data.index)
    _sub_with_last = pd.Series(np.zeros(data.shape[0]),index = data.index)
    _tag_feature = pd.Series(np.zeros(data.shape[0]),index = data.index)
    user_dup_dict = {}
    print("begin produce dict with {(userID,creativeID):[clickTime_1,clickTime_2,…,clickTime_last]}")
    print("----------------------------------------------------------------------------------------")
    num =0
    for index in ix:
        num += 1
        if num % print_size == 0:
            print("the data size that have been handled is {0}".format(num))
        u = ur[index] # userID
        k = ck[index] # clickTime
        e = ce[index] # creativeID
        ue = (u,e)  #(userID,creativeID)
        k_list = [k]
        if ue not in user_dup_dict: 
            user_dup_dict.setdefault(ue,k_list) 
        else: 
            k_dup_list = user_dup_dict.get(ue) 
            k_dup_list.append(k)
            user_dup_dict[ue] = k_dup_list
    print("finish prouce dict in time {0:6.0f}".format(time.time()-_time))
    print("sort the values in the dict")
    _time = time.time()
    keys = user_dup_dict.keys()
    for key in keys:
        user_dup_dict.get(key).sort()
    print("finish sort in time {0:6.0f}".format(time.time()-_time))
    print("----------------------------------------------------")
    print("begin produce feature")
    num = 0
    for i_1 in ix:
        num += 1
        if num % print_size == 0:
            print("the data size that have been handled is {0}".format(num))
        u1 = ur[i_1] # userID
        k1 = ck[i_1] # clickTime
        e1 = ce[i_1] # creativeID
        ue1 = (u1,e1)  #(userID,creativeID)
        click_list = user_dup_dict.get(ue1)
        #标记特征
        if len(click_list) == 1:
            _tag_feature[i_1] = 0
        elif k1 == click_list[0] :
            _tag_feature[i_1] = 1
        elif k1 == click_list[len(click_list)-1] :
            _tag_feature[i_1] = 2
        else:
            _tag_feature[i_1] = 3
        # 当前记录与重复记录的时间差特征
        sub_with_first = k1 - click_list[0]
        _log_sub = math.log1p(sub_with_first + 1)
        _log_sub = int(_log_sub)
        _sub_with_first[i_1] = _log_sub
        sub_with_last = click_list[len(click_list)-1] - k1
        _log_last = math.log1p(sub_with_last + 1)
        _log_last = int(_log_last)
        _sub_with_last[i_1] = _log_last 
    print("finsh produce feature in time {0:6.0f}".format(time.time()-_totalTime))
    return _tag_feature,_sub_with_first,_sub_with_last


def get_dupTime_fea2(df,user = 'userID',creative = 'creativeID',click = 'clickTime',print_size = 1000000):
    _time = time.time()
    _totalTime = _time
    now = _totalTime
    df = df.sort_values(by=['userID','creativeID','clickTime'])
    data = df[[user,creative,click]]
    ur = data[user]
    ce = data[creative]
    ck = data[click]
    ix = list(data.index)
    _sub_with_first = pd.Series(np.zeros(data.shape[0]),index = data.index)
    _sub_with_last = pd.Series(np.zeros(data.shape[0]),index = data.index)
    _tag_feature = pd.Series(np.zeros(data.shape[0]),index = data.index)
    ur_array = np.zeros(data.shape[0])
    ce_array = np.zeros(data.shape[0])
    ck_array = np.zeros(data.shape[0])
    print("begin array")
    num = 0
    for i in ix:
        num += 1
        if num % print_size ==0:
             print("the data size that have been handled is {0}".format(num))
        ur_array[i] = ur[i]
        ce_array[i] = ce[i]
        ck_array[i] = ck[i]
    print("create array finished in time {0:6.0f}".format(time.time()-now))
    now = time.time()
    print("---------------------------------------------------------------")
    print("begin reconde last index")
    last_index = []
    num = 0
    for list_index in range(0,len(ix)): 
        num += 1
        if num % print_size ==0:
             print("the data size that have been handled is {0}".format(num))
        ue_cur = (ur_array[ix[list_index]],ce_array[ix[list_index]]) 
        ue_pre = (ur_array[ix[list_index - 1]],ce_array[ix[list_index - 1]])
        try:
            ue_pos = (ur_array[ix[list_index + 1]],ce_array[ix[list_index + 1]])
        except:
            ue_pos = -1
        if ue_cur != ue_pre and ue_cur != ue_pos: # 不重复
            last_index.append(ix[list_index])
       # elif ue_cur != ue_pre and ue_cur == ue_pos: # 重复记录第一条 
        elif ue_cur == ue_pre and ue_cur != ue_pos: # 重复记录最后一条
            last_index.append(ix[list_index])
   #     elif ue_cur == ue_pre and ue_cur == ue_pos: # 重复记录既不是第一条又不是最后一条 
    print("finish reconde last index in time {0:6.0f}".format(time.time()-now))
    print("---------------------------------------------------------------")
    print("begin handle data")
    fit_inx_red = 0
    num = 0
    for list_index in range(0,len(ix)):
        num += 1
        if num % print_size ==0:
             print("the data size that have been handled is {0}".format(num)) 
       # print(num)
        ue_cur = (ur_array[ix[list_index]],ce_array[ix[list_index]]) 
        ue_pre = (ur_array[ix[list_index - 1]],ce_array[ix[list_index - 1]])
        try:
            ue_pos = (ur_array[ix[list_index + 1]],ce_array[ix[list_index + 1]])
        except:
            ue_pos = -1
        if ue_cur != ue_pre and ue_cur != ue_pos:
            _tag_feature[ix[list_index]] = 0  # 不重复
            _log_sub_first = math.log10(ck_array[ix[list_index]] + 1)
            _sub_with_first[ix[list_index]]  = _log_sub_first
            if ix[list_index] != last_index[0]:
                sub_last = ck_array[last_index[0]] - ck_array[ix[list_index]] 
                _log_sub_last = math.log10(sub_last + 1)
                _sub_with_last[ix[list_index]] = _log_sub_last
            else:
                sub_last = ck_array[last_index[0]] - ck_array[ix[list_index]] 
                _log_sub_last = math.log10(sub_last + 1)
                _sub_with_last[ix[list_index]] = _log_sub_last
                try:
                    last_index.remove(last_index[0])
                except:
                    continue
        elif ue_cur != ue_pre and ue_cur == ue_pos:
            _tag_feature[ix[list_index]] = 1  # 重复记录第一条 
            fit_inx_red = ix[list_index] # 记录重复数据第一条的索引
            sub = ck_array[ix[list_index]] - ck_array[fit_inx_red]
            _log_sub = math.log10(sub + 1) 
            _sub_with_first[ix[list_index]] = _log_sub
            if ix[list_index] != last_index[0]:
                sub_last = ck_array[last_index[0]] - ck_array[ix[list_index]] 
                _log_sub_last = math.log10(sub_last + 1)
                _sub_with_last[ix[list_index]] = _log_sub_last
            else:
                sub_last = ck_array[last_index[0]] - ck_array[ix[list_index]] 
                _log_sub_last = math.log10(sub_last + 1)
                _sub_with_last[ix[list_index]] = _log_sub_last
                try:
                    last_index.remove(last_index[0])
                except:
                    continue
        elif ue_cur == ue_pre and ue_cur != ue_pos:
            _tag_feature[ix[list_index]] = 2  # 重复记录最后一条
            sub = ck_array[ix[list_index]] - ck_array[fit_inx_red]
            _log_sub = math.log10(sub + 1) 
            _sub_with_first[ix[list_index]] = _log_sub
            if ix[list_index] != last_index[0]:
                sub_last = ck_array[last_index[0]] - ck_array[ix[list_index]] 
                _log_sub_last = math.log10(sub_last + 1)
                _sub_with_last[ix[list_index]] = _log_sub_last
            else:
                sub_last = ck_array[last_index[0]] - ck_array[ix[list_index]] 
                _log_sub_last = math.log10(sub_last + 1)
                _sub_with_last[ix[list_index]] = _log_sub_last
                try:
                    last_index.remove(last_index[0])
                except:
                    continue
        elif ue_cur == ue_pre and ue_cur == ue_pos:
            _tag_feature[ix[list_index]] = 3 # 重复记录既不是第一条又不是最后一条
            sub = ck_array[ix[list_index]] - ck_array[fit_inx_red]
            _log_sub = math.log10(sub + 1) 
            _sub_with_first[ix[list_index]] = _log_sub
            if ix[list_index] != last_index[0]:
                sub_last = ck_array[last_index[0]] - ck_array[ix[list_index]] 
                _log_sub_last = math.log10(sub_last + 1)
                _sub_with_last[ix[list_index]] = _log_sub_last
            else:
                sub_last = ck_array[last_index[0]] - ck_array[ix[list_index]] 
                _log_sub_last = math.log10(sub_last + 1)
                _sub_with_last[ix[list_index]] = _log_sub_last
                try:
                    last_index.remove(last_index[0])
                except:
                    continue
    print("finish produce feature in time {0:6.0f}".format(time.time()-_totalTime))
    return _tag_feature,_sub_with_first,_sub_with_last
train = pd.read_csv(r'E:\finaldata\final\train.csv')
test = pd.read_csv(r'E:\finaldata\final\test.csv')
data = pd.concat([train,test],ignore_index=True)
data['day'] = np.round(data.clickTime / 1000000)
del train
del test
gc.collect()
data['user_dup_tag_feature'],data['click_sub_with_first'],data['click_sub_with_last'] = get_dupTime_fea(data,print_size=5000000)      
data = data[['label','day','userID','user_dup_tag_feature','click_sub_with_first','click_sub_with_last']]
data.to_csv(r'E:\finaldata\user_tag1.csv',index = False)