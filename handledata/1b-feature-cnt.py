import numpy as np 
import pandas as pd 
import utils
from utils import *


data = pd.read_csv(utils.data_path + "/data.csv")
print ("finished loading raw data, ", data.shape)


print ("to add some basic features ...")
data['day'] = np.round(data.clickTime / 1000000)
#data['weekday'] = np.round((data.day - 17) % 7)
data['hour'] = np.round(data.clickTime % 1000000 / 10000)
data['day_hour'] = (data.day.values - 26) * 24 + data.hour.values
data = data[data.day>27]
# del org
# rank特征
# 可以用所有的上下文特征
print ("count the order ...")
data['rank_userID'] = my_grp_idx(data.userID.values.astype('str'), data.instanceID.values.astype('str'))
data['rank_day_userID'] = my_grp_idx(np.add(data.userID.values.astype('str'), data.day.astype('str').values).astype('str'), data.instanceID.values.astype('str'))
data['rank_hour_userID'] = my_grp_idx(np.add(data.userID.values.astype('str'), data.day_hour.astype('str').values).astype('str'), data.instanceID.values.astype('str'))
data['rank_creative_userID'] = my_grp_idx(np.add(data.userID.values.astype('str'), data.creativeID.astype("str").values).astype('str'), data.instanceID.values.astype('str'))
data['rank_app_userID'] = my_grp_idx(np.add(data.userID.values.astype('str'), data.appID.astype("str").values).astype('str'), data.instanceID.values.astype('str'))
data['rank_positionID_userID'] = my_grp_idx(np.add(data.userID.values.astype('str'), data.positionID.astype('str').values).astype('str'), data.instanceID.values.astype('str'))
# data['rank_connectionType_userID'] = my_grp_idx(np.add(data.userID.values.astype('str'), data.connectionType.astype('str').values).astype('str'), data.instanceID.values.astype('str'))
# data['rank_positionType_userID'] = my_grp_idx(np.add(data.userID.values.astype('str'), data.positionType.astype('str').values).astype('str'), data.instanceID.values.astype('str'))

# count
print ("count ...")
data['cnt_userID'] = get_agg(data.userID.values.astype('str'), data.instanceID.values.astype('str'),np.size)
data['cnt_cam'] = get_agg(data.camgaignID.values.astype('str'), data.instanceID.values.astype('str'),np.size)

print ("trick feature ...")
data['userclickTimeDup'] = data.duplicated(['userID', 'clickTime'], keep = False).map(lambda x: 1 if x == True else 0)
dup = data[data['userclickTimeDup'] == 1].copy()
dup['whetherFistRecord'] = dup.duplicated(['userID', 'clickTime'], keep = 'first').map(lambda x: 2 if x == True else 1)
data.loc[data.userclickTimeDup == 1, 'userclickTimeDup'] = dup['whetherFistRecord']
del dup

# count unique
print ("count unique feature...")
# data['campaignID_cnt_by_advertiserID'] = my_grp_cnt(data.advertiserID.values.astype('str'), data.camgaignID.values.astype('str'))
# data['adID_cnt_by_campaignID'] = my_grp_cnt(data.camgaignID.values.astype('str'), data.adID.values.astype('str'))
# data['creativeID_cnt_by_adID'] = my_grp_cnt(data.adID.values.astype('str'), data.creativeID.values.astype('str'))
# data['campaignID_cnt_by_appID'] = my_grp_cnt(data.appID.values.astype('str'), data.camgaignID.values.astype('str'))
# data['adID_cnt_by_appID'] = my_grp_cnt(data.appID.values.astype('str'), data.adID.values.astype('str'))
# data['appID_cnt_by_hour'] = my_grp_cnt(data.hour.values.astype('str'), data.appID.values.astype('str'))
data['creativeID_cnt_by_appID'] = my_grp_cnt(data.appID.values.astype('str'), data.creativeID.values.astype('str'))

data['creative_cnt_by_user_id'] = my_grp_cnt(data.userID.values.astype('str'), data.creativeID.values.astype('str'))
data['creative_cnt_by_hour_user_id'] = my_grp_cnt(np.add(data.userID.astype('str').values, data.day_hour.astype('str').values), data.creativeID.values.astype('str'))
data['creative_cnt_by_day_user_id'] = my_grp_cnt(np.add(data.userID.astype('str').values, data.day.astype('str').values), data.creativeID.values.astype('str'))
data['day_cnt_by_user_id'] = my_grp_cnt(data.userID.astype('str').values, data.day.values.astype('str'))
data['app_cnt_by_user_id'] = my_grp_cnt(data.userID.values.astype('str'), data.appID.values.astype('str'))
data['app_cnt_by_hour_user_id'] = my_grp_cnt(np.add(data.userID.astype('str').values, data.day_hour.astype('str').values), data.appID.values.astype('str'))
data['app_cnt_by_day_user_id'] = my_grp_cnt(np.add(data.userID.astype('str').values, data.day.astype('str').values), data.appID.values.astype('str'))

data.to_csv(utils.data_path + 'data_cnt_0612.csv',index=False,encoding='utf-8')