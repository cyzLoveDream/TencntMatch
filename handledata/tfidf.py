# -*- coding: utf-8 -*-
"""
Created on Tue May 30 12:26:49 2017

@author: Administrator
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv(r'E:\tencent\pre\train1.csv')
test = pd.read_csv(r'E:\tencent\pre\test1.csv')

user_installedapps = pd.read_csv(r'E:\tencent\pre\user_installed_apps.csv')
print(1)
user_installedapps = user_installedapps.groupby('userID').agg(lambda x:' '.join(['app'+str(s) for s in x.values])).reset_index()

user_id_all = pd.concat([train.userID,test.userID],axis=0)
user_id_all = pd.DataFrame(user_id_all,columns=['userID'])

user_installedapps = pd.merge(user_id_all.drop_duplicates(),user_installedapps,on='userID',how='left')
user_installedapps = user_installedapps.fillna('Missing')

tfv = TfidfVectorizer()
tfv.fit(user_installedapps.appID)
print(2)
user_installedapps = pd.merge(user_id_all,user_installedapps,on='userID',how='left')
user_installedapps = user_installedapps.fillna('Missing')
print(3)
user_installedapps_tfv = tfv.transform(user_installedapps.appID)
print(4)