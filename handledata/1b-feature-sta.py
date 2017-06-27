import numpy as np 
import pandas as pd 
import utils
from utils import *
import gc

# org = pd.read_csv('../test/data_test.csv')
org = pd.read_csv('../data/data_2829.csv')
''' 这里主要计算排序和点击率，点击率排行信息，基础偏差　加上　用户偏差
    userID adID appID Context(包括时间)
    排序:使用 cyz的代码，有问题
    点击率： owen zhang的代码
'''
# --------------------------------------
# basic time 
start_day = 27
# data['day'] = np.round(data.clickTime / 1000000)
#data['weekday'] = np.round((data.day - 17) % 7)
org['hour'] = np.round(org.clickTime % 1000000 / 10000)
org['day_hour'] = (org.day.values - start_day) * 24 + org.hour.values

fea_name = ['userID', 'camgaignID',
       'creativeID',  'positionID', 'day','day_hour','label','instanceID',
       'adID', 'advertiserID',
       'appID', 'appCategory']
data = org[fea_name]
del org 
gc.collect()

print('-------------advertiserID 与 campaignID---------------------')
# advertiserID 与 campaignID
#　单特征点击数，转化数计算
data['cnt_by_advertiserID'] = my_grp_cnt(data.advertiserID.values.astype('str'), data.instanceID.values.astype('str'))
data['con_cnt_advertiserID'] = my_grp_label_cnt(data.advertiserID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
#　求交叉特征
data['advertiserID_campaignID'] = np.add(data.advertiserID.astype('str').values,data.camgaignID.astype('str').values)
# 交叉特征点击数，转化数计算
data['con_cnt_advertiserID_campaignID'] = my_grp_label_cnt(data.advertiserID_campaignID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
data['cnt_by_advertiserID_campaignID'] = my_grp_cnt(data.advertiserID_campaignID.values.astype('str'),data.instanceID.values.astype('str'))
# # 交叉特征点击率，转化率计算
print('-------------campaignID & adID---------------------')
# campaignID & adID
#　单特征点击数，转化数计算
data['cnt_by_camgaignID'] = my_grp_cnt(data.camgaignID.values.astype('str'), data.instanceID.values.astype('str'))
data['con_cnt_camgaignID'] = my_grp_label_cnt(data.camgaignID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
#　求交叉特征
data['adID_campaignID'] = np.add(data.adID.astype('str').values,data.camgaignID.astype('str').values)
# 交叉特征点击数，转化数计算
data['con_cnt_adID_campaignID'] = my_grp_label_cnt(data.adID_campaignID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
data['cnt_by_adID_campaignID'] = my_grp_cnt(data.adID_campaignID.values.astype('str'),data.instanceID.values.astype('str'))

print('-------------adID & creativeID---------------------')

# adID & creativeID
data['cnt_by_adID'] = my_grp_cnt(data.adID.values.astype('str'), data.instanceID.values.astype('str'))
data['con_cnt_adID'] = my_grp_label_cnt(data.adID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
#　求交叉特征
data['adID_creativeID'] = np.add(data.adID.astype('str').values,data.creativeID.astype('str').values)
# 交叉特征点击数，转化数计算
data['con_cnt_adID_creativeID'] = my_grp_label_cnt(data.adID_creativeID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
data['cnt_by_adID_creativeID'] = my_grp_cnt(data.adID_creativeID.values.astype('str'),data.instanceID.values.astype('str'))


print('-------------reativeID & positionID---------------------')

# creativeID & positionID
data['cnt_by_creativeID'] = my_grp_cnt(data.creativeID.values.astype('str'), data.instanceID.values.astype('str'))
data['con_cnt_creativeID'] = my_grp_label_cnt(data.creativeID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
#　求交叉特征
data['positionID_creativeID'] = np.add(data.positionID.astype('str').values,data.creativeID.astype('str').values)
# 交叉特征点击数，转化数计算
data['con_cnt_positionID_creativeID'] = my_grp_label_cnt(data.positionID_creativeID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
data['cnt_by_positionID_creativeID'] = my_grp_cnt(data.positionID_creativeID.values.astype('str'),data.instanceID.values.astype('str'))
# appID
# appID adID
print('-------------adID & creativeID---------------------')

# adID & creativeID
data['cnt_by_appID'] = my_grp_cnt(data.appID.values.astype('str'), data.instanceID.values.astype('str'))
data['con_cnt_appID'] = my_grp_label_cnt(data.appID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
#　求交叉特征
data['adID_appID'] = np.add(data.adID.astype('str').values,data.appID.astype('str').values)
# 交叉特征点击数，转化数计算
data['con_cnt_adID_appID'] = my_grp_label_cnt(data.adID_appID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
data['cnt_by_adID_appID'] = my_grp_cnt(data.adID_appID.values.astype('str'),data.instanceID.values.astype('str'))

print('-------------adID appID---------------------')

# adID appID
data['cnt_by_appID'] = my_grp_cnt(data.appID.values.astype('str'), data.instanceID.values.astype('str'))
data['con_cnt_appID'] = my_grp_label_cnt(data.appID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
#　求交叉特征
data['adID_appID'] = np.add(data.adID.astype('str').values,data.appID.astype('str').values)
# 交叉特征点击数，转化数计算
data['con_cnt_adID_appID'] = my_grp_label_cnt(data.adID_appID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
data['cnt_by_adID_appID'] = my_grp_cnt(data.adID_appID.values.astype('str'),data.instanceID.values.astype('str'))

data['cnt_by_userID'] = my_grp_cnt(data.userID.values.astype('str'), data.instanceID.values.astype('str'))
data['con_cnt_userID'] = my_grp_label_cnt(data.userID.values.astype('str'), data.instanceID.values.astype('str'),data.label)

print('-------------userID appCategory---------------------')

# userID appCategory
#　求交叉特征
data['appCategory_userID'] = np.add(data.appCategory.astype('str').values,data.userID.astype('str').values)
# 交叉特征点击数，转化数计算
data['con_cnt_appCategory_userID'] = my_grp_label_cnt(data.appCategory_userID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
data['cnt_by_appCategory_userID'] = my_grp_cnt(data.appCategory_userID.values.astype('str'),data.instanceID.values.astype('str'))

print('-------------userID appID---------------------')

# userID appID
#　求交叉特征
data['appID_userID'] = np.add(data.appID.astype('str').values,data.userID.astype('str').values)
# 交叉特征点击数，转化数计算
data['con_cnt_appID_userID'] = my_grp_label_cnt(data.appID_userID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
data['cnt_by_appID_userID'] = my_grp_cnt(data.appID_userID.values.astype('str'),data.instanceID.values.astype('str'))
# 交叉特征点击率，转化率计算
print('------------userID adID---------------------')

# userID adID
#　求交叉特征
data['adID_userID'] = np.add(data.adID.astype('str').values,data.userID.astype('str').values)
# 交叉特征点击数，转化数计算
data['con_cnt_adID_userID'] = my_grp_label_cnt(data.adID_userID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
data['cnt_by_adID_userID'] = my_grp_cnt(data.adID_userID.values.astype('str'),data.instanceID.values.astype('str'))
print('-------------userID creativeID---------------------')

# userID creativeID
data['cnt_by_userID'] = my_grp_cnt(data.userID.values.astype('str'), data.instanceID.values.astype('str'))
data['con_cnt_userID'] = my_grp_label_cnt(data.userID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
#　求交叉特征
data['creativeID_userID'] = np.add(data.adID.astype('str').values,data.userID.astype('str').values)
# 交叉特征点击数，转化数计算
data['con_cnt_creativeID_userID'] = my_grp_label_cnt(data.creativeID_userID.values.astype('str'), data.instanceID.values.astype('str'),data.label)
data['cnt_by_creativeID_userID'] = my_grp_cnt(data.creativeID_userID.values.astype('str'),data.instanceID.values.astype('str'))

# The count unique feature
print('-------------userID & adID,appID,creativeIDD---------------------')

# userID & adID,appID,creativeID
data['appID_cnt_by_userID'] = my_grp_cnt(data.userID.values.astype('str'), data.appID.values.astype('str'))
data['appCategory_cnt_by_userID'] = my_grp_cnt(data.userID.values.astype('str'), data.appCategory.values.astype('str'))
data['adID_cnt_by_userID'] = my_grp_cnt(data.userID.values.astype('str'), data.adID.values.astype('str'))
data['creativeID_cnt_by_userID'] = my_grp_cnt(data.userID.values.astype('str'), data.creativeID.values.astype('str'))
# ad相关
data['campaignID_cnt_by_advertiserID'] = my_grp_cnt(data.advertiserID.values.astype('str'), data.camgaignID.values.astype('str'))
data['adID_cnt_by_campaignID'] = my_grp_cnt(data.camgaignID.values.astype('str'), data.adID.values.astype('str'))
data['creativeID_cnt_by_adID'] = my_grp_cnt(data.adID.values.astype('str'), data.creativeID.values.astype('str'))
data['positionID_cnt_by_creativeID'] = my_grp_cnt(data.creativeID.values.astype('str'), data.positionID.values.astype('str'))
# appID adID
data['adID_cnt_by_appID'] = my_grp_cnt(data.appID.values.astype('str'), data.adID.values.astype('str'))
# appID,adID 的热度
data['userID_cnt_by_appID'] = my_grp_cnt(data.appID.values.astype('str'), data.userID.values.astype('str'))
data['userID_cnt_by_adID'] = my_grp_cnt(data.adID.values.astype('str'), data.userID.values.astype('str'))

print('--------Saving data---------')
data.to_csv('../data/data_feature_0618_part1.csv',index=False,encoding='utf-8')

