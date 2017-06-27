import pandas as pd
import numpy as np
from joblib import dump, load, Parallel, delayed
import utils
from utils import *


data_path = 'D:/MyDeveloperWay/Kaggle/pre/data'


df_train = pd.read_csv(data_path + '/train.csv')
df_ad = pd.read_csv(data_path +'/ad.csv')
df_app_categories = pd.read_csv(data_path +'/app_categories.csv')
df_position = pd.read_csv(data_path +'/position.csv')
df_user = pd.read_csv(data_path +'/user.csv')
df_test = pd.read_csv(data_path +'/test.csv')
if utils.sample_pct < 1.0:
	np.random.seed(999)
	r1 = np.random.uniform(0, 1, df_train.shape[0])
	df_train = df_train.ix[r1 < utils.sample_pct, :]
	print("testing with small sample of training data, ", df_train.shape)
df_train['instanceID'] = range(1+df_test.shape[0],df_train.shape[0]+df_test.shape[0]+1)
df_all = pd.concat([df_train,df_test])



print("Merging all the data......")
df_all = pd.merge(df_all,df_ad,on='creativeID')
df_all = pd.merge(df_all,df_position,on='positionID')
df_all = pd.merge(df_all,df_user,on='userID')
df_all = pd.merge(df_all,df_app_categories,on='appID')

df_all.to_csv(utils.temp_data_path+'/df_all_merge.csv',encoding='utf-8')
print ("finished loading raw data, ", df_all.shape)

print("add some basic features ...")
df_all['day']=np.round(df_all.clickTime / 10000)
df_all['hour'] = np.round(df_all.clickTime % 10000 / 100)
df_all['minutes'] = np.round(df_all.clickTime % 100)
df_all['day_hour'] = (df_all.day.values - 17) * 24 + df_all.hour.values
df_all['day_hour_prev'] = df_all['day_hour'] - 1
df_all['day_hour_next'] = df_all['day_hour'] + 1
df_all['day_minutes'] = df_all.day_hour.values * 60 + df_all.minutes.values
df_all['day_minutes_prev'] = df_all['day_minutes'] - 1
df_all['day_minutes_next'] = df_all['day_minutes'] + 1

print("to rank userID by ip ...")
df_all['rank_userID'] = my_grp_idx(df_all.userID.values.astype('str'), df_all.clickTime.values.astype('str'))
df_all['rank_day_userID'] = my_grp_idx(np.add(df_all.userID.astype('str').values, df_all.day.astype('str').values).astype('str'), df_all.clickTime.values.astype('str'))
df_all['rank_app_userID'] = my_grp_idx(np.add(df_all.userID.astype('str').values, df_all.appID.astype('str').values).astype('str'), df_all.clickTime.values.astype('str'))

print("add some count features ...")
#统计广告属性，每个广告主有几个推广计划，每个推广计划有几个广告，每个广告对应多少素材
df_all['campaignID_cnt_by_advertiserID'] = my_grp_cnt(df_all.advertiserID.values.astype('str'), df_all.camgaignID.values.astype('str'))
df_all['adID_cnt_by_campaignID'] = my_grp_cnt(df_all.camgaignID.values.astype('str'), df_all.adID.values.astype('str'))
df_all['creativeID_cnt_by_adID'] = my_grp_cnt(df_all.adID.values.astype('str'), df_all.creativeID.values.astype('str'))


#利用app作为媒介，统计app对应的广告个数，每个app对应几个推广计划，每个app对应几个广告
df_all['campaignID_cnt_by_appID'] = my_grp_cnt(df_all.appID.values.astype('str'), df_all.camgaignID.values.astype('str'))
df_all['adID_cnt_by_appID'] = my_grp_cnt(df_all.appID.values.astype('str'), df_all.adID.values.astype('str'))

df_all['userID_cnt'] = get_agg(df_all.userID.values,df_all.instanceID.values, np.size)

print("to count prev/current/next hour/minutes by ip ...")
cntDualKey(df_all, 'userID', None, 'day_hour', 'day_hour_prev', fill_na=0)
cntDualKey(df_all, 'userID', None, 'day_hour', 'day_hour', fill_na=0)
cntDualKey(df_all, 'userID', None, 'day_hour', 'day_hour_next', fill_na=0)

cntDualKey(df_all, 'userID', None, 'day_minutes', 'day_minutes_prev', fill_na=0)
cntDualKey(df_all, 'userID', None, 'day_minutes', 'day_minutes', fill_na=0)
cntDualKey(df_all, 'userID', None, 'day_minutes', 'day_minutes_next', fill_na=0)

df_all['pday'] = df_all.day - 1
calcDualKey(df_all, 'userID', None, 'day', 'pday', 'label', 10, None, True, True)

df_all['cnt_diff_duserID_day_pday'] = df_all.cnt_userID_day.values  - df_all.cnt_userID_pday.values
df_all['diff_cnt_dev_ip_hour_phour_prev'] = (df_all.cnt_userID_day_hour.values - df_all.cnt_userID_day_hour_prev.values)
df_all['diff_cnt_dev_ip_hour_phour_next'] = (df_all.cnt_userID_day_hour.values - df_all.cnt_userID_day_hour_next.values) 
df_all['diff_cnt_dev_ip_minutes_pminutes_prev'] = (df_all.cnt_userID_day_minutes.values - df_all.cnt_userID_day_minutes_prev.values)
df_all['diff_cnt_dev_ip_minutes_pminutes_next'] = (df_all.cnt_userID_day_minutes.values - df_all.cnt_userID_day_minutes_next.values)



print("to generate t0tv_mx .. ")
list_param = [ 'instanceID','label','age','gender','education','marriageStatus','haveBaby','hometown','residence','sitesetID', 'positionType','connectionType','telecomsOperator']
feature_list_name = 'cnt'
feature_list_dict = {}

feature_list_dict[feature_list_name] = list_param + \
['campaignID_cnt_by_advertiserID', 'adID_cnt_by_campaignID',
       'creativeID_cnt_by_adID', 'campaignID_cnt_by_appID','adID_cnt_by_appID','userID_cnt',
       'cnt_diff_duserID_day_pday','diff_cnt_dev_ip_hour_phour_prev',
       'diff_cnt_dev_ip_hour_phour_next','diff_cnt_dev_ip_minutes_pminutes_prev',
       'diff_cnt_dev_ip_minutes_pminutes_next' ,'rank_userID','rank_day_userID','rank_app_userID']


#_start_day = 18 
#filter_tv = np.logical_and(df_all.day.values >= _start_day, df_all.day.values < 31)
#filter_t1 = np.logical_and(df_all.day.values < 30, filter_tv)
#filter_v1 = np.logical_and(~filter_t1, filter_tv)    
    
#t0tv_mx = df_all.as_matrix(feature_list_dict[feature_list_name])
# t0tv_mx = df_all[feature_list_dict[feature_list_name]]

print("to save t0tv_mx ...")

# t0tv_mx = df_all[feature_list_dict[feature_list_name]]
# t0tv_mx.to_csv(utils.temp_data_path + '/t0tv_mx_all_cnt_0523_0914.csv',encoding='utf_8')
df_all.to_csv(utils.temp_data_path + '/df_all_0523_0914.csv',encoding='utf_8')
#t0tv_mx_save = {}
#t0tv_mx_save['t0tv_mx'] = t0tv_mx
#t0tv_mx_save['label'] = df_all.label.values
#t0tv_mx_save['day'] = df_all.day.values
#t0tv_mx_save['instanceID'] = df_all.adID.values
#if utils.sample_pct <1:
    #dump(t0tv_mx_save, utils.temp_data_path + '/t0tv_mx_0523.joblib_dat')
#else:
    #dump(t0tv_mx_save, utils.temp_data_path + '/t0tv_mx_all_0523.joblib_dat')


