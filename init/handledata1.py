# -*- coding: utf-8 -*-
"""
Created on Sat May 13 08:49:40 2017

@author: Administrator
"""

import pandas as pd

trains = pd.read_csv(r'E:\tencent\input\trainresultuse.csv')
#tests = pd.read_csv(r'E:\tencent\pre\testresult.csv')
addata = pd.read_csv(r'E:\tencent\pre\ad.csv')
appdata = pd.read_csv(r'E:\tencent\pre\app_categories.csv')

#加入广告特征
addata = addata.drop('advertiserID',axis=1)
addata = addata.drop('appID',axis = 1)
addata = addata.drop('appPlatform',axis = 1)
trains = trains.merge(addata, how='left', on='creativeID')
#tests = tests.merge(addata, how='left', on='creativeID')

#加入APP特征
trains = trains.merge(appdata,how='left',on='appID')
#tests = tests.merge(appdata,how='left',on='appID')
#tests.to_csv(r'E:\tencent\allmerge\testresult.csv',index = False)

uapp = pd.read_csv(r'E:\tencent\input\uapp1.csv')
trains = trains.merge(uapp,how='left',on='userID')
em = pd.read_csv(r'E:\tencent\input\egmh.txt')
trains = trains.merge(em,how='left',on='appCategory')
trains.to_pickle(r'E:\tencent\input\trainresult')

del addata
del appdata
del trains
 