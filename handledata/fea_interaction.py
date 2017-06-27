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
'''
if len(sys.argv) == 1:
	sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())
'''

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



#pick only 23 - 31 days data
# 数据采样
'''
day_test = 31
sample_pct = 0.1
if sample_pct <1.0:
    np.random.seed(999)
    df_train = data[data['day']<day_test]
    r1 = np.random.uniform(0,1,df_train.shape[0])
    df_train = df_train.ix[r1<sample_pct,:]
    data = pd.concat([df_train,data[data['day']==day_test]])
'''
'''  
data = data[data['day'] != 17]
data = data[data['day'] != 18]
data = data[data['day'] != 19]
data = data[data['day'] != 20]
data = data[data['day'] != 21]
'''
fea_data = data
del data
gc.collect()
print("testing with small sample of training data 2 , ",fea_data.shape) 

# 点击率特征
fea_lists = ['positionID_advertiserID', 'positionID_appID', 'positionID_positionType', 'positionID_camgaignID', 'positionID_connectionType', 'positionID_adID', 'appID_connectionType', 'advertiserID_positionType', 'appID_positionType', 'positionID_creativeID',  'creativeID_positionType',  'creativeID_gender', 'creativeID_age']

for fea in fea_lists:
    print("current feature is ", fea)
    fea_1 = fea.split('_')[0]
    fea_2 = fea.split('_')[1]
    print ('fea_1:{0}, fea_2:{1}'.format(fea_1, fea_2))
    fea_data[fea] = pd.Series(np.add(fea_data[fea_1].astype('str').values , fea_data[fea_2].astype('str').values)).astype('category').values.codes
	
calc_exptv(fea_data,  ['positionID_advertiserID'])
del fea_lists[0]
calc_exptv(fea_data, fea_lists)
calc_exptv(fea_data, ['positionID_advertiserID'], add_count=True)

single_fea_lists = ['positionID', 'age', 'creativeID', 'advertiserID', 'camgaignID', 'appID', 'hometown', 'hour', 'adID', 'residence']
calc_exptv(fea_data, single_fea_lists)


print ('to encode categorical features using mean responses from earlier days -- multivariate')
vns = ['positionID_advertiserID', 'positionID_appID', 'positionID_positionType', 'positionID_camgaignID', 'positionID_connectionType', 'positionID_adID', 'appID_connectionType', 'advertiserID_positionType', 'appID_positionType', 'positionID_creativeID',  'creativeID_positionType',  'creativeID_gender', 'creativeID_age']

#dftv = data.ix[np.logical_and(data.day.values >= 17, data.day.values < 32), ['label', 'day', 'id'] + vns].copy()
dftv = fea_data.ix[np.logical_and(fea_data.day.values >= 27, fea_data.day.values < 32), ['label', 'day', 'id'] + vns].copy()
for vn in vns:
    dftv[vn] = dftv[vn].astype('category')
    print (vn)

#n_ks = {'positionID_advertiserID', 'positionID_appID', 'positionID_positionType', 'positionID_sitesetID', 'positionID_camgaignID', 'positionID_connectionType', 'advertiserID_sitesetID', 'positionID_adID', 'appID_connectionType', 'positionID_gender', 'advertiserID_positionType', 'creativeID_camgaignID', 'appID_positionType', 'positionID_creativeID', 'creativeID_appID',  'creativeID_positionType', 'appID_gender', 'camgaignID_connectionType',  'creativeID_gender', 'creativeID_age'}
exp2_dict = {}
for vn in vns:
    exp2_dict[vn] = np.zeros(dftv.shape[0])

days_npa = dftv.day.values

#for day_v in xrange(18, 32):
for day_v in range(28, 32):
    df1 = dftv.ix[np.logical_and(dftv.day.values < day_v, dftv.day.values < 31), :].copy()
    df2 = dftv.ix[dftv.day.values == day_v, :]
    print ("Validation day:", day_v, ", train data shape:", df1.shape, ", validation data shape:", df2.shape)
    pred_prev = df1.label.values.mean() * np.ones(df1.shape[0])
    for vn in vns:
        if 'exp2_'+vn in df1.columns:
            df1.drop('exp2_'+vn, inplace=True, axis=1)
    for i in range(3):
        for vn in vns:
            #p1 = calcLeaveOneOut2(df1, vn, 'label', n_ks[vn], 0, 0.25, mean0=pred_prev)
            p1 = calcLeaveOneOut2(df1, vn, 'label', 100, 0, 0.25, mean0=pred_prev)
            pred = pred_prev * p1
            print (day_v, i, vn, "change = ", ((pred - pred_prev)**2).mean())
            pred_prev = pred
        del pred
        gc.collect() 
        pred1 = df1.label.values.mean()
        for vn in vns:
            print ("="*20, "merge", day_v, vn)
            diff1 = mergeLeaveOneOut2(df1, df2, vn)
            pred1 *= diff1
            exp2_dict[vn][days_npa == day_v] = diff1

        pred1 *= df1.label.values.mean() / pred1.mean()
        print ("logloss = ", logloss(pred1, df2.label.values))
    del df1
    del df2
    gc.collect()

for vn in vns:
    fea_data['exp2_'+vn] = exp2_dict[vn]
    

#trick feature
print ('to count prev/current/next hour by appID ...')
feature_list = ['appID','userID','creativeID','positionID','adID','sitesetID','advertiserID']
for feature in feature_list:
    cntDualKey(fea_data,feature,None,'day_hour','day_hour_prev',fill_na=0)
    cntDualKey(fea_data,feature,None,'day_hour','day_hour',fill_na=0)
    cntDualKey(fea_data,feature,None,'day_hour','day_hour_next',fill_na=0)
    print('feature : %s is done',feature)



# 计数特征，这里应该还可以优化
print ('count feature...')
fea_data['campaignID_cnt_by_advertiserID'] = my_grp_cnt(fea_data.advertiserID.values.astype('str'), fea_data.camgaignID.values.astype('str'))
fea_data['adID_cnt_by_campaignID'] = my_grp_cnt(fea_data.camgaignID.values.astype('str'), fea_data.adID.values.astype('str'))
fea_data['creativeID_cnt_by_adID'] = my_grp_cnt(fea_data.adID.values.astype('str'), fea_data.creativeID.values.astype('str'))
fea_data['campaignID_cnt_by_appID'] = my_grp_cnt(fea_data.appID.values.astype('str'), fea_data.camgaignID.values.astype('str'))
fea_data['adID_cnt_by_appID'] = my_grp_cnt(fea_data.appID.values.astype('str'), fea_data.adID.values.astype('str'))
fea_data['appID_cnt_by_hour'] = my_grp_cnt(fea_data.hour.values.astype('str'), fea_data.appID.values.astype('str'))

print("the data size is {0},{1}".format(fea_data.shape[0],fea_data.shape[1]))
test = fea_data[fea_data['day']==31]
train = fea_data[fea_data['day']!=31]
test.to_pickle(r'E:\finaldata\final\sample\feadata\test')
train.to_pickle(r'E:\finaldata\final\sample\feadata\train')
#data.to_pickle(r'E:\tencent\input\nofeature\alldatafeaturenext')
print("finish all")
