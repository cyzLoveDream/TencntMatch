# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 10:07:38 2017

@author: adc
"""

import pandas as pd
import numpy as np
import time
import argparse, csv, sys
#import cPickle 
import gc
from utils import my_grp_idx
from utils import calc_exptv
from utils import calcLeaveOneOut2 
from utils import mergeLeaveOneOut2
from utils import my_grp_cnt
from utils import logloss
from utils import cntDualKey

fea_data = pd.read_pickle(r'E:\finaldata\final\sample\feadata\rank_trick')

print("testing with small sample of training data 2 , ",fea_data.shape)  

#trick feature
print ('to count prev/current/next hour by appID ...')
feature_list = ['appID','userID','creativeID','positionID','adID','sitesetID','advertiserID']
for feature in feature_list:
    cntDualKey(fea_data,feature,None,'day_hour','day_hour_prev',fill_na=0)
    cntDualKey(fea_data,feature,None,'day_hour','day_hour',fill_na=0)
    cntDualKey(fea_data,feature,None,'day_hour','day_hour_next',fill_na=0)
    print('feature : %s is done',feature)
del feature_list
gc.collect()
# 计数特征，这里应该还可以优化
print ('count feature...')
fea_data['campaignID_cnt_by_advertiserID'] = my_grp_cnt(fea_data.advertiserID.values.astype('str'), fea_data.camgaignID.values.astype('str'))
fea_data['adID_cnt_by_campaignID'] = my_grp_cnt(fea_data.camgaignID.values.astype('str'), fea_data.adID.values.astype('str'))
fea_data['creativeID_cnt_by_adID'] = my_grp_cnt(fea_data.adID.values.astype('str'), fea_data.creativeID.values.astype('str'))
fea_data['campaignID_cnt_by_appID'] = my_grp_cnt(fea_data.appID.values.astype('str'), fea_data.camgaignID.values.astype('str'))
fea_data['adID_cnt_by_appID'] = my_grp_cnt(fea_data.appID.values.astype('str'), fea_data.adID.values.astype('str'))
fea_data['appID_cnt_by_hour'] = my_grp_cnt(fea_data.hour.values.astype('str'), fea_data.appID.values.astype('str'))
fea_data['creativeID_cnt_by_appID'] = my_grp_cnt(fea_data.appID.values.astype('str'), fea_data.creativeID.values.astype('str')) 
fea_data['creative_cnt_by_user_id'] = my_grp_cnt(fea_data.userID.values.astype('str'), fea_data.creativeID.values.astype('str'))
fea_data['creative_cnt_by_hour_user_id'] = my_grp_cnt(np.add(fea_data.userID.astype('str').values, fea_data.day_hour.astype('str').values), fea_data.creativeID.values.astype('str'))
fea_data['creative_cnt_by_day_user_id'] = my_grp_cnt(np.add(fea_data.userID.astype('str').values, fea_data.day.astype('str').values), fea_data.creativeID.values.astype('str'))
fea_data['day_cnt_by_user_id'] = my_grp_cnt(fea_data.userID.astype('str').values, fea_data.day.values.astype('str'))
fea_data['app_cnt_by_user_id'] = my_grp_cnt(fea_data.userID.values.astype('str'), fea_data.appID.values.astype('str'))
fea_data['app_cnt_by_hour_user_id'] = my_grp_cnt(np.add(fea_data.userID.astype('str').values, fea_data.day_hour.astype('str').values), fea_data.appID.values.astype('str'))
fea_data['app_cnt_by_day_user_id'] = my_grp_cnt(np.add(fea_data.userID.astype('str').values, fea_data.day.astype('str').values), fea_data.appID.values.astype('str'))
print("the data size is {0},{1}".format(fea_data.shape[0],fea_data.shape[1]))
test = fea_data[fea_data['day']==31]
train = fea_data[fea_data['day']!=31]
del fea_data
gc.collect()
test.to_pickle(r'E:\finaldata\final\sample\feadata\test') 
del test 
gc.collect()
train.to_pickle(r'E:\finaldata\final\sample\feadata\train')
del train
gc.collect()
#data.to_pickle(r'E:\tencent\input\nofeature\alldatafeaturenext')
print("finish all")
 
