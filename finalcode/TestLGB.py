from TencentMathc.finalcode import esemble
import pandas as pd
import  gc

data = pd.read_csv(r'E:\finaldata\final\sample\feadata\data.csv')
train_data =data[data['day']<=30]
test_data = data[data['day']==31]
del data
gc.collect()
print(1)
#处理空值
test_data = test_data.fillna(value=0)
train_data = train_data.fillna(value=0)
print("data input finish in size")
features = [x for x in train_data.columns if
            x not in ['label', 'instanceID', 'day', 'clickTime', 'userID', 'conversionTime', '_bindApp_fea']]
val_data = train_data[train_data['day'] == 30]
train_data = train_data[train_data['day'] != 30]
#获取测试集

print(4)
#设置参数
parms = {
     'boosting_type': 'gbdt',
    # 这里二分类，是一个二类的问题，因此采用了binary:logistic分类器，
    'objective': 'binary',
	'metric': 'binary_logloss',
	'num_leaves': 31,
	'learning_rate': 0.05,
	'feature_fraction': 0.9,
	'bagging_fraction': 0.8,
	'bagging_freq': 5,
	'verbose': 0
	
    # 'gamma': 0.2,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    # 'scale_pos_weight':1,
    # 'lambda':2,
    # 'max_depth': 10,  # 构建树的深度 [1:] 最佳
    # 'lambda':450,  # L2 正则项权重
    # 'subsample': 1,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    # 'colsample_bytree': 0.7,  # 构建树树时的采样比率 (0:1]
   # 'min_child_weight':8, # 节点的最少特征数
    # 'silent': 1,
    # 'eta': 0.1,  # 如同学习率
    # 'seed': 780,
    # 'nthread': 7,
    # 'max_delta_step':6,
    # 'min_child_weight': 3, #最佳
    # 'metric':'logloss',
    # 'n_estimators':200
}

lgb = esemble.LGB(parms,500,features)
lgb.fit(train_data,val_data)
lgb.predict(test_data)







