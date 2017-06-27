import pandas as pd
import numpy as np
import utils

df_all = pd.read_csv(utils.temp_data_path+'/df_all_exp_cnt_app_0530.csv')

print(df_all.shape)
#The time interval from the last click
def cal_click_diff(df):
    st = pd.Series(170000)
    instanceID = df['instanceID']
    df = df['clickTime']
    df = df.append(st).sort_values()
    diff = pd.DataFrame({'instanceID':instanceID,'clickTime':df[1:].values,'click_interval':df[1:].values - df[:-1].values})
    return diff

print('----------The time interval from the last click----------')
user_group = df_all.groupby('userID')
user_click_diff = user_group['clickTime','instanceID'].apply(lambda x: cal_click_diff(x))
df_all = pd.merge(df_all,user_click_diff[['instanceID','click_interval']],on='instanceID')
print(df_all[df_all.day==31].shape)
print(df_all.shape)
#　分割特征
print('-----------Make the category feature to be seperated-----------')
df_all['appCat_1'] = df_all['appCategory'].map(lambda x:round(x/100))
df_all['appCat_2'] = df_all['appCategory'].map(lambda x:round(x%100))

df_all['home_province'] = df_all['hometown'].map(lambda x:round(x/100))
df_all['home_city'] = df_all['hometown'].map(lambda x:round(x%100))
df_all['live_province'] = df_all['residence'].map(lambda x:round(x/100))
df_all['live_city'] = df_all['residence'].map(lambda x:round(x%100))

df_all = df_all.drop(['appCategory','hometown','residence'],axis = 1)
print(df_all[df_all.day==31].shape)
#　特征融合

#　计数特征
# Number of impressions to the user for the ad　id/ad group id in the hour/day
def fea_to_user_cnt_day(df_all,fea_list):
    for fea in fea_list:
        key = fea + '_to_user_count_day'
        for day in range(17,32):
            df = df_all.loc[df_all['day']==day,[fea,'userID','instanceID']].copy()
            grp = df.groupby([fea,'userID'])
            size = grp['instanceID'].size().reset_index()
            size.columns = [fea,'userID',key]
            df = pd.merge(df,size,on=[fea,'userID'])
            print(df.columns)
            df_all.loc[df_all['day']==day,key] = df[key].values

def fea_to_user_cnt_hour(df_all,fea_list):
    for fea in fea_list:
        key = fea + '_to_user_count_hour'
        for hour in range(0,25):
            df = df_all.loc[df_all['hour']==hour,[fea,'userID','instanceID']].copy()
            grp = df.groupby([fea,'userID'])
            size = grp['instanceID'].size().reset_index()
            size.columns = [fea,'userID',key]
            df = pd.merge(df,size,on=[fea,'userID'])
            df_all.loc[df_all['hour']==hour,key] = df[key].values

print('-------Caculating the user click for a specific adID in day/ hour-----------')
fea_to_user_cnt_list = ['adID','creativeID','advertiserID', 'camgaignID','appID']
fea_to_user_cnt_day(df_all,fea_to_user_cnt_list)
print(df_all[df_all.day==31].shape)
fea_to_user_cnt_hour(df_all,fea_to_user_cnt_list)
print(df_all[df_all.day==31].shape)

#The number of days user appeared
print('-----Calculating The number of days user appeared-----')
user_group = df_all.groupby('userID')
user_first_time = user_group['clickTime'].apply(lambda x:int(np.sort(x)[0]/10000)).reset_index()
user_first_time.columns= ['userID','first_appear']
df_all = pd.merge(df_all,user_first_time,on='userID')
df_all['duration'] = df_all['day'] - df_all['first_appear']
print(df_all[df_all.day==31].shape)

#t0tv_mx.to_csv(utils.temp_data_path + '/t0tv_mx_all_exp_0527.csv',index=False; encoding='utf_8')
df_all.to_csv(utils.temp_data_path + '/df_all_cnt_exp_0601.csv',index=False,encoding='utf_8')
#t0tv_mx_save = {}
#t0tv_mx_save['t0tv_mx'] = t0tv_mx
#t0tv_mx_save['label'] = df_all.label.values
#t0tv_mx_save['day'] = df_all.day.values
#t0tv_mx_save['instanceID'] = df_all.adID.values
#if utils.sample_pct <1:
    #dump(t0tv_mx_save, utils.temp_data_path + '/t0tv_mx_0523.joblib_dat')
#else:
    #dump(t0tv_mx_save, utils.temp_data_path + '/t0tv_mx_all_0523.joblib_dat')
