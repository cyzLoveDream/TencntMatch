# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:09:57 2017

@author: Administrator
"""

import pandas as pd
import xgboost as xgb 
import gc
import time

nowall = time.time()
now = nowall
train_data = pd.read_pickle(r'E:\tencentfinal\final\final\sample\feadata\train')
test_data = pd.read_pickle(r'E:\tencentfinal\final\final\sample\feadata\test')
print("finish read data ,the traindata size is {0},{1}.the test data size is {2},{3}".format(train_data.shape[0],train_data.shape[1],test_data.shape[0],test_data.shape[1])) 
test_data = test_data.fillna(value=0)
train_data = train_data.fillna(value=0) 
def modelfit(alg,dtrain,dtest,features,target='label',useTrainCV = True,cv_folds = 5,early_stopping_rounds=50):
    dtrains = dtrain
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[features].values, label=dtrain[target].values,feature_names=features) 
        del dtrain
        gc.collect()
        print("begin cv")
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='logloss', early_stopping_rounds=early_stopping_rounds, show_stdv=False,seed=904,verbose_eval=50)
        alg.set_params(n_estimators=cvresult.shape[0])
    print("cv finish")
    print(cvresult)
    val_data = dtrains[dtrains['day']==30]
    dtrains = dtrains[dtrains['day']!=30]
     #Fit the algorithm on the data
    alg.fit(dtrains[features], dtrains[target],eval_set=[(dtrains[features],dtrains[target]),(val_data[features],val_data[target])],eval_metric='neg_logloss',verbose=True)
    evals_result = alg.evals_result()
    print(evals_result)
    del dtrains
    gc.collect()
    #Predict test set: 
    dtest_pred_prob = alg.predict_proba(dtest[features])[:,1]
    print("finish pred mean :{0}".format(dtest_pred_prob.mean()))
    p=pd.DataFrame({"increaseid":range(1,1+len(dtest_pred_prob)),"prob":dtest_pred_prob})
    p.to_csv(r'E:\tencentfinal\final\result\submission.csv',index=False)
    cost_time = time.time() - nowall
    print("end ......", '\n', "cost time:", cost_time, "(s)......")
features = [x for x in train_data.columns if x not in ['label','instanceID','day','clickTime','userID','conversionTime']]
parms = {
    'booster': 'gbtree',
    # 这里二分类，是一个二类的问题，因此采用了binary:logistic分类器，
    'objective': 'binary:logistic',
    'gamma': 0.2,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'scale_pos_weight':10,
    'lambda':2,
    'max_depth': 10,  # 构建树的深度 [1:] 最佳
    # 'lambda':450,  # L2 正则项权重
    'subsample': 1,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_bytree': 0.7,  # 构建树树时的采样比率 (0:1]
   # 'min_child_weight':8, # 节点的最少特征数 
    'silent': 1,
    'eta': 0.1,  # 如同学习率
    'seed': 780,
    'nthread': 7,
    'max_delta_step':6,
    'min_child_weight': 3, #最佳
    'eval_metric':'logloss',
    'n_estimators':200
}
xgb1 =xgb.XGBClassifier(parms)
modelfit(xgb1,train_data,test_data,features)

