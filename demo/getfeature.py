# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:49:42 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
import os

train = pd.read_pickle(r'E:\tencent\input\nofeature\trainresult')
test = pd.read_pickle(r'E:\tencent\input\nofeature\test') 
train = train[train['clickTime']<270000]
#user = pd.read_csv(r'E:\tencent\pre\user.csv') 
#positionID,connectionType,telecomsOperator,advertiserID,appPlatform,sitesetID,positionType,camgaignID

ui = ['positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
       'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence',
       'advertiserID', 'appID', 'appPlatform', 'sitesetID', 'positionType',
       'adID']
col = [x for x  in ui]
for i in col:
    m = train.groupby([i,'label']).size()
    m = pd.DataFrame(m)
    m = m.unstack()
   # u = train.groupby(i).size()
  #  m = pd.concat([m,u],axis=1)
    m.columns = ['0','1']
  #  t = np.array(m['all'])
   # m = m.div(t,axis=0)
  #  out = pd.DataFrame(m)
    m.to_csv(r'E:\tencent\1\\'+i+'.txt')
    
'''
i=0
for filename in os.listdir(r'E:\tencent\trainFeature'):
    i +=1
    path = r'E:\\tencent\trainFeature\\' + filename
    data = pd.read_csv(path)
    index = filename.index('.')
    name = filename[:index]
    train = train.merge(data,how = 'left',on=name)
    test = test.merge(data,how='left',on = name)
    train = train.drop(name,axis = 1)
    test = test.drop(name,axis = 1)
    print(i)
'''