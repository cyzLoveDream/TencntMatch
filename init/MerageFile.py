# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 22:09:12 2017

@author: Administrator
"""

import pandas as pd
import numpy as np

gender=pd.read_csv(r'E:\tencent\trainFeature\gender.csv')
eduAndMs =pd.read_csv(r'E:\tencent\trainFeature\eduAndMs.csv')
#hb = pd.read_csv(r'E:\tencent\trainFeature\hb.csv')
#ht = pd.read_csv(r'E:\tencent\trainFeature\ht.csv')
#age = pd.read_csv(r'E:\tencent\trainFeature\age.csv')
user=pd.read_csv(r'E:\tencent\trainFeature\user.csv')

user = user.iloc[:,1:]
gender = gender.iloc[:,1:]
gender = gender[['userID','label','prex1','prex2','prex3']]
gender = gender.drop('label',axis =1)
eduAndMs = eduAndMs.iloc[:,1:]
eduAndMs = eduAndMs[['userID','label','prex4','prex5','prex6','prex7','prex8','prex9','prex10','prex11','prex12','prex13','prex14','prex15']]
eduAndMs=eduAndMs.drop('label',axis=1)
#hb = hb.iloc[:,1:]
#ht = ht.iloc[:,1:]
#age = age.iloc[:,1:]

user = user.merge(gender,on='userID',how='left')
user=user.merge(eduAndMs,on='userID',how='left')
#data = data.merge(hb,on='userID',how='left')
#data = data.merge(ht,on='userID',how='left')
#data = data.merge(age,on='userID',how='left')
# data.to_csv(r'E:\tencent\trainFeature\user.csv')