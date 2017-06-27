import pandas as pd
import numpy as np
import time,csv,hashlib
import math


org = pd.read_csv(r'E:\finaldata\final\sample\feadata\totalData.csv')
m = pd.read_csv(r'E:\finaldata\user_tag_result.csv')
org = org[['adID', 'advertiserID', 'age', 'appCategory', 'appID',
       'appPlatform', 'camgaignID', 'clickTime', 'connectionType',
       'conversionTime', 'creativeID', 'day', 'education', 'gender',
       'haveBaby', 'hometown', 'instanceID', 'label', 'marriageStatus',
       'positionID', 'positionType', 'residence', 'sitesetID',
       'telecomsOperator', 'userID', 'cnt_cre_positionID',
       'cnt_cov_crepositionID', 'cnt_cre_connectionType',
       'cnt_cov_creconnectionType', 'cnt_cre_telecomsOperator',
       'cnt_cov_cretelecomsOperator', 'cnt_cre_gender', 'cnt_cov_cregender',
       'cnt_cre_education', 'cnt_cov_creeducation', 'cnt_cre_marriageStatus',
       'cnt_cov_cremarriageStatus', 'cnt_cre_haveBaby', 'cnt_cov_crehaveBaby',
       'cnt_cre_hometown', 'cnt_cov_crehometown', 'cnt_cre_sitesetID',
       'cnt_cov_cresitesetID', 'cnt_cre_positionType',
       'cnt_cov_crepositionType', '_bindApp_fea', 'cnt_positionID',
       'cnt_cov_positionID', 'cnt_connectionType', 'cnt_cov_connectionType',
       'cnt_telecomsOperator', 'cnt_cov_telecomsOperator', 'cnt_hometown',
       'cnt_cov_hometown', 'cnt_sitesetID', 'cnt_cov_sitesetID',
       'cnt_positionType', 'cnt_cov_positionType']]



org = org.merge(m,how='left',on='userID')
del m 
print(time.time())
print("org loaded with shape", org.shape)
print(org.columns.values)

# 类别特征
fea_list_cat = ['adID', 'advertiserID', 'age', 'appCategory', 'appID',
       'appPlatform', 'camgaignID','education', 'gender',
       'haveBaby', 'hometown','marriageStatus',
       'positionID', 'positionType', 'residence', 'sitesetID',
       'telecomsOperator','user_dup_tag_feature']
# 统计特征
fea_list_num = ['cnt_cre_positionID',
       'cnt_cov_crepositionID', 'cnt_cre_connectionType',
       'cnt_cov_creconnectionType', 'cnt_cre_telecomsOperator',
       'cnt_cov_cretelecomsOperator', 'cnt_cre_gender', 'cnt_cov_cregender',
       'cnt_cre_education', 'cnt_cov_creeducation', 'cnt_cre_marriageStatus',
       'cnt_cov_cremarriageStatus', 'cnt_cre_haveBaby', 'cnt_cov_crehaveBaby',
       'cnt_cre_hometown', 'cnt_cov_crehometown', 'cnt_cre_sitesetID',
       'cnt_cov_cresitesetID', 'cnt_cre_positionType',
       'cnt_cov_crepositionType', '_bindApp_fea', 'cnt_positionID',
       'cnt_cov_positionID', 'cnt_connectionType', 'cnt_cov_connectionType',
       'cnt_telecomsOperator', 'cnt_cov_telecomsOperator', 'cnt_hometown',
       'cnt_cov_hometown', 'cnt_sitesetID', 'cnt_cov_sitesetID',
       'cnt_positionType', 'cnt_cov_positionType','click_sub_with_first', 'click_sub_with_last']
# 转化率特征
fea_list_cor = []

fea_list = fea_list_cat + fea_list_num

data = org.ix[:, ['label','instanceID'] + fea_list_cor ].copy()
idx_base = 2

for vn in fea_list:
    if vn in fea_list_num:
        _cat = pd.Series(np.log(org[vn].values)).astype('category').values.codes
    else:
        _cat = org[vn].astype('category').values.codes       
    _cat1 = _cat + idx_base
    data[vn] = _cat1
    print(vn, idx_base, _cat1.min(), _cat1.max(), np.unique(_cat).size)
    idx_base += _cat.max() + 1
    
print  ("to save data ...")
data.to_csv(r'E:\finaldata\ffm\data_ffm.csv',index=False,encoding='utf-8')

train_out = 'E:\\finaldata\\ffm\\train.ffm'
valid_out = 'E:\\finaldata\\ffm\\valid.ffm'
test_out = 'E:\\finaldata\\ffm\\test.ffm'

def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf-8')).hexdigest(), 16)%(nr_bins-1)+1

start = time.time()
count = 0

train_output = open(train_out, 'w')
valid_output = open(valid_out, 'w') 
test_output = open(test_out, 'w') 

nr_bins = int(2e+6)

data = '../data/data_ffm.csv'
fea_list = fea_list_cat + fea_list_num + fea_list_cor

for row in csv.DictReader(open(data)):
    count += 1
    if count%1000000 == 0:
        print(count,str(time.time()-start))
    feats = []
    i = 0
    for fea in fea_list:
        i += 1
        # 点击率特征和其他特征转化格式不一样
        if fea in fea_list_cor:
            feats.append('{0}:{1}:{2}'.format(i, idx_base+i, row[fea]))
        else:
            feats.append('{0}:{1}:1'.format(i, hashstr(row[fea], nr_bins)))

    if int(float(row['day'])) < 30: 
        train_output.write(row['label'] + ' ' + ' '.join(feats) + '\n')
    elif float(row['day']) == 30:
        valid_output.write(row['label'] + ' ' + ' '.join(feats) + '\n')
    else:
        test_output.write(row['label'] + ' ' + ' '.join(feats) + '\n')

train_output.close()
valid_output.close()
test_output.close()
