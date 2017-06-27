# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:33:52 2017

@author: Administrator
"""

import xgboost as xgb
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import os
rcParams['figure.figsize'] = 12, 4
        
def ceate_feature_map(features):  
    outfile = open(r'E:\tencent\model\xgb.fmap', 'w')  
    i = 0  
    for feat in features:  
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1  
    outfile.close() 
    
train_data = pd.read_pickle(r'E:\tencent\input\noonehot\train')
i=0
for file in os.listdir(r'E:\tencent\trainfeaturecount'):
    i+=1
    filename = "E:\\tencent\\trainfeaturecount\\"+file
    index = file.index('.')
    f = file[:index]
    data = pd.read_csv(filename)
    train_data = train_data.merge(data,how='left',on=f)
  #  test_data = test_data.merge(data,how='left',on=f)
    print(i)
#labels = train_data.iloc[:,:1].values
#trains = train_data.iloc[:,3:]
train_data['appCategory_1'] = train_data['appCategory'].map(lambda x: round(x / 100))   # 列处理
train_data['appCategory_2'] = train_data['appCategory'].map(lambda x: x % 100)
train_data['home_province'] = train_data['hometown'].map(lambda x: round(x / 100))
train_data['home_city'] = train_data['hometown'].map(lambda x: x % 100)
train_data['live_province'] = train_data['residence'].map(lambda x: round(x / 100))
train_data['live_city'] = train_data['residence'].map(lambda x: x % 100)
#处理test
features = [x for x in train_data.columns if x not in ['label','conversionTime']]
ceate_feature_map(features=features)
model = xgb.Booster(model_file=r'E:\tencent\model\modeltrain1046')

feat_imp = pd.Series(model.get_fscore(fmap=r'E:\tencent\model\xgb.fmap')).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')