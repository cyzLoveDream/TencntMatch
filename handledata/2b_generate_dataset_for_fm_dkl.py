import pandas as pd
import numpy as np
import time,csv,hashlib
import utils
from utils import *


org = pd.read_csv('../data/data.csv')

print(time.time())
print("org loaded with shape", org.shape)
print(org.columns.values)

# 类别特征
fea_list_cat = []
# 统计特征
fea_list_num = []
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
data.to_csv('../data/data_ffm.csv',index=False,encoding='utf-8')

train_out = '../data/train.ffm'
valid_out = '../data/valid.ffm'
test_out = '../data/test.ffm'

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
