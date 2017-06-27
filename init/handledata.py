# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:22:04 2017

@author: Administrator
"""

import pandas as pd

import time 

now = time.time()
test_data = pd.read_csv(r'E:\tencent\pre\test.csv')
train_data = pd.read_csv(r'E:\tencent\pre\train.csv')
user_data = pd.read_csv(r'E:\tencent\pre\user.csv')
ad_data = pd.read_csv(r'E:\tencent\pre\ad.csv')
position_data = pd.read_csv(r'E:\tencent\pre\position.csv')
# 加入用户信息特征
train_data = train_data.merge(user_data, how='left', on='userID')
test_data = test_data.merge(user_data, how='left', on='userID')

# 加入广告特征
ad_data = ad_data.drop('adID', axis=1)
ad_data = ad_data.drop('camgaignID', axis=1)
train_data = train_data.merge(ad_data, how='left', on='creativeID')
test_data = test_data.merge(ad_data, how='left', on='creativeID')

# 加入位置特征
train_data = train_data.merge(position_data, how='left', on='positionID')
test_data = test_data.merge(position_data, how='left', on='positionID')

del user_data
del ad_data
del position_data