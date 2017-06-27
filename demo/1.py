# -*- coding: utf-8 -*-
"""
Created on Sat May 13 23:05:32 2017

@author: Administrator
"""

import xgboost as xgb

dtrain = xgb.DMatrix(r'E:\tencent\output\train.txt')
dtest = xgb.DMatrix(r'E:\tencent\output\dtest.txt')
param = {'max_depth':6, 'eta':0.3, 'silent':1, 'objective':'binary:logistic' }  
  
# specify validations set to watch performance  
watchlist  = [(dtest,'eval'), (dtrain,'train')]  
num_round = 20  
bst = xgb.train(param, dtrain, num_round, watchlist)  


