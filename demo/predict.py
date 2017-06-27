# -*- coding: utf-8 -*-
"""
Created on Sun May 21 10:55:01 2017

@author: Administrator
"""

import pandas as pd
import xgboost as xgb
import time
import os
import gc
now = time.time()

data = pd.read_csv(r'E:\finaldata\final\sample\feadata\data.csv') 
test_data = data[data['day']==31]
 
print(0)
i=0
def ceate_feature_map(features):  
    outfile = open(r'E:\finaldata\model\xgb.fmap', 'w')  
    i = 0  
    for feat in features:  
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1  
    outfile.close() 
print('先分类')
'''
test_data = test_data.fillna(value=0)
test_data['appCategory_app'] = test_data['appCategory'].map(lambda x: round(x / 100))   # 列处理
test_data['appCategory_appc'] = test_data['appCategory'].map(lambda x: x % 100)
test_data['home_province'] = test_data['hometown'].map(lambda x: round(x / 100))
test_data['home_city'] = test_data['hometown'].map(lambda x: x % 100)
test_data['live_province'] = test_data['residence'].map(lambda x: round(x / 100))
test_data['live_city'] = test_data['residence'].map(lambda x: x % 100)
test_data['age_sex'] = test_data['age'] + test_data['gender']*100
#时间特征
print("begin train time feature")
test_data['hour'] = (test_data['clickTime']-test_data['day']*1000000).map(lambda x: int( x/10000))
test_data['minute'] =(test_data['clickTime']-(test_data['day']*1000000+test_data['hour']*10000)).map(lambda x: int (x/1000))
test_data['day_hour'] = (test_data['day']*1000000 + test_data['hour']*10000).map(lambda x: int(x))
test_data['hour_minute'] = test_data['hour']*1000 + test_data['minute']
test_data['pre_day'] = test_data['day']-1
test_data['age_split'] = test_data['age'].map(lambda x: int(x / 5))
print("begin train time feature")
'''
'''
print('合特征文件1')
for file in os.listdir(r'E:\tencent\1'):
    i+=1
    filename = "E:\\tencent\\1\\"+file
    index = file.index('.')
    f = file[:index]
    data = pd.read_csv(filename)
   # train_data = train_data.merge(data,how='left',on=f)
    test_data = test_data.merge(data,how='left',on=f)
    print(i)

print('合特征文件2')
i=0
i=0
for file in os.listdir(r'E:\tencent\trainfeaturedoublecount'):
    i+=1
    fn = "E:\\tencent\\trainfeaturedoublecount\\"+file
    index = file.index('.')
    f = file[:index]
    data = pd.read_csv(fn)
    test_data = test_data.merge(data,how='left',on=f)
  #  test_data = test_data.merge(data,how='left',on=f)
    print(i)
''' 
#test_data = test_data.drop('userID',axis=1)
features = [x for x in test_data.columns if x not in ['label','instanceID','day','clickTime','userID','conversionTime']]
tests = test_data[features]
xgTest = xgb.DMatrix(tests) 
del test_data
gc.collect() 
print(1)
model = xgb.Booster(model_file=r'E:\finaldata\model\modeltrain')
print(2) 
pre = model.predict(xgTest)
print("finish pred mean {0}".format(pre.mean()))
print(3)
#xgb.plot_importance(model)
p=pd.DataFrame({"increaseid":range(1,1+len(pre)),"prob":pre}) 
p.to_csv(r'E:\finaldata\result\submission.csv',index=False)

cost_time = time.time() - now
print("end ......", '\n', "cost time:", cost_time, "(s)......")
