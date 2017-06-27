

import pandas as pd
import  numpy as np
import  time

#以下数据测试使用
now = time.time()
train = pd.read_pickle(r'E:\tencent\input\rawdata\train')
test = pd.read_pickle(r'E:\tencent\input\rawdata\test')
addata = pd.read_csv(r'E:\tencent\pre\ad.csv')
test['label'] = 0
train.drop('conversionTime', axis = 1, inplace = True)
test.drop('instanceID', axis = 1, inplace = True)
data = pd.concat([train, test])
#加入广告特征
addata = addata.drop('advertiserID',axis=1)
addata = addata.drop('appID',axis = 1)
addata = addata.drop('appPlatform',axis = 1)
data = data.merge(addata, how='left', on='creativeID')

appCate = pd.read_csv(r'E:\tencent\pre\app_categories.csv')
data = data.merge(appCate,how = 'left',on='appID')
data['day'] = np.round(data.clickTime / 10000)

print("finish input rawdata in {0}".format(time.time()-now))
now = time.time()
uapp = pd.read_csv(r'E:\tencent\importantfeature\uapp1.csv')
uapp.fillna(value=0,inplace=True)
user =pd.read_csv(r'E:\tencent\pre\user.csv')
print("finish input user app install in {0}".format(time.time()-now))
now = time.time()
#构建原始历史用户安装记录DataFrame
ones = pd.DataFrame(np.zeros((user.shape[0],0)),index=range(1,user.shape[0]+1))
ones['userID'] = range(1,user.shape[0]+1)
ones = ones.merge(uapp,how='left',on='userID')
ones.fillna(value=0,inplace=True)
ones.drop('userID',axis=1,inplace=True)
ones['101'] = 0
ones.columns = ones.columns.astype('int')
print("finish create DataFrame in {0}".format(time.time()-now))
data_result = pd.DataFrame()
for day in range(17,32):
    print("begin handle {0} day ".format(day))
    now = time.time()
    data_day = data[data['day']==day]
    for i in data_day.index.values:
        _data = data_day.ix[i]
        if _data['label'] == 1:
            _user_in = _data['userID'].astype('int')
            _appCate_col = _data['appCategory'].astype('int')
            ones.at[_user_in+1,_appCate_col] += 1
    ones['userID'] = ones.index + 1
    data_day = data_day.merge(ones,how = 'left',on='userID')
    print("finish handle {0} day in {1}".format(day,time.time()-now))
    data_result = pd.concat([data_result,data_day])
    ones.drop('userID',axis = 1,inplace=True)
print("begin write data to model")
now = time.time()
data_result[data_result['day']<31].to_pickle(r'E:\tencent\input\lr\train')
data_result[data_result['day']==31].to_pickle(r'E:\tencent\input\lr\test')
print("finish write data in {0}".format(time.time()-now))
