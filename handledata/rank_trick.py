# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:52:50 2017

@author: adc
""" 
import pandas as pd
import numpy as np

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

def ageDiscretization(x):
    if x == 0:
        return 0
    elif x >= 1 and x <= 9:
        return 1
    elif x >= 10 and x <= 17:
        return 2
    elif x >= 18 and x <= 27:
        return 3
    elif x >= 28 and x <= 38:
        return 4
    elif x >= 39 and x <= 49:
        return 5
    elif x >= 50 and x <= 53:
        return 6
    else:
        return 7
train = pd.read_pickle(r'E:\finaldata\final\sample\rawdata\train')
#train = train.sample(frac=0.2)
test = pd.read_pickle(r'E:\finaldata\final\sample\rawdata\test')
test['label'] = 0
train.drop('conversionTime', axis = 1, inplace = True)
test.drop('instanceID', axis = 1, inplace = True)

data = pd.concat([train, test]) 
print ('finished loading raw data, ', data.shape)
del train
del test
gc.collect()


print('splite some feature from hometown and province and appCategory')
'''
data['appCategory_app'] = data['appCategory'].map(lambda x: round(x / 100))   # 列处理
data['appCategory_appc'] = data['appCategory'].map(lambda x: x % 100)
data['home_province'] = data['hometown'].map(lambda x: round(x / 100))
data['home_city'] = data['hometown'].map(lambda x: x % 100)
data['live_province'] = data['residence'].map(lambda x: round(x / 100))
data['live_city'] = data['residence'].map(lambda x: x % 100)
data['age_sex'] = data['age'] + data['gender']*100
'''

print ('to add some basic features ...')
#data['day'] = np.round(data.clickTime / 10000)
#data['weekday'] = np.round((data.day - 17) % 7)
data['hour'] = np.round(data.clickTime % 1000000 / 10000)
data['day_hour'] = (data.day.values - 17) * 24 + data.hour.values
data['day_hour_prev'] = data['day_hour'] - 1
data['day_hour_next'] = data['day_hour'] + 1
#data['residence_province'] = np.round(data['residence'] / 100)
data['minute'] =(data['clickTime']-(data['day']*1000000+data['hour']*10000)).map(lambda x: int (x/1000))
data['splite_age'] = data['age'].map(lambda x: ageDiscretization(x))

 
# rank特征
print ('count the order ...')
data['rank_userID'] = my_grp_idx(data.userID.values.astype('str'), data.creativeID.values.astype('str'))
data['rank_day_userID'] = my_grp_idx(np.add(data.userID.values.astype('str'), data.day.astype('str').values).astype('str'), data.creativeID.values.astype('str'))
data['rank_sitesetID_userID'] = my_grp_idx(np.add(data.userID.values.astype('str'), data.sitesetID.astype('str').values).astype('str'), data.creativeID.values.astype('str'))
data['rank_positionID_userID'] = my_grp_idx(np.add(data.userID.values.astype('str'), data.positionID.astype('str').values).astype('str'), data.creativeID.values.astype('str'))
data['rank_adID_userID'] = my_grp_idx(np.add(data.userID.values.astype('str'),data.adID.astype('str').values).astype('str'),data.creativeID.values.astype('str'))

# trick feature
data['userclickTimeDup'] = data.duplicated(['userID', 'clickTime'], keep = False).map(lambda x: 1 if x == True else 0)
dup = data[data['userclickTimeDup'] == 1].copy()
dup['whetherFistRecord'] = dup.duplicated(['userID', 'clickTime'], keep = 'first').map(lambda x: 2 if x == True else 1)
data.loc[data.userclickTimeDup == 1, 'userclickTimeDup'] = dup['whetherFistRecord']

print("the data size is {0} * {1}".format(data.shape[0],data.shape[1]))
data.to_pickle(r'E:\finaldata\final\sample\feadata\rank_trick')
