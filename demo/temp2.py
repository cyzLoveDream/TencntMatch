# Import libraries
import pandas as pd
import numpy as np
import xgboost as xgb

from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation,metrics
from sklearn.grid_search import GridSearchCV
import time

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,4

def modelfit(alg,dtrain,predictors,useTrainCV = True,cv_folds = 5,early_stopping_rounds = 50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values,label = dtrain[target].values)
        cvresult = xgb.cv(xgb_param,xgtrain,num_boost_round = alg.get_params()['n_estimators'],nfold = cv_folds,
                          metrics = 'logloss',early_stopping_rounds = early_stopping_rounds,show_stdv = False)
        alg.set_params(n_estimators = cvresult.shape[0])

        #Fit the algorithm on the data
        alg.fit(dtrain[predictors],dtrain['label'],eval_metric = 'logloss')

        #Predict training set:
      #  dtrain_predictions = alg.predict(dtrain[predictors])
       # dtrain_prob = alg.predict_proba(dtrain[predictors])[:,1]

        #print model report：
        print('\nModel Report')
        evals_result = alg.evals_result()
        print("logloss %.4g" %  evals_result)
       # print("AUC Score(train)：%f" % metrics.roc_auc_score(dtrain['label'],dtrain_prob))

        xgb.plot_importance(alg)

now = time.time() 
train_data = pd.read_pickle(r'E:\tencent\input\train')
print(1)
test_data = pd.read_pickle(r'E:\tencent\input\test')
print(2) 
#处理空值
test_data = test_data.fillna(value=0)
train_data = train_data.fillna(value=0)
 
#获取训练集L

#trains = train_data.drop('f1') 
now = time.time()
# 进行Onehot
#处理空值
test_data = test_data.fillna(value=0)
train_data = train_data.fillna(value=0)
target='label'
conver =  'clickTime' 
click = 'conversionTime'
predictor = [x for x in train_data.columns if x not in [target,conver,click]]
xgb1 = XGBClassifier(
   
    # 这里二分类，是一个二类的问题，因此采用了binary:logistic分类器，
    objective= 'binary:logistic',
    gamma = 0.1,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    scale_pos_weight=1,
    reg_lambda=2,
    max_depth= 10,  # 构建树的深度 [1:] 最佳
    # 'lambda':450,  # L2 正则项权重
    subsample= 0.8,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    colsample_bytree= 0.7,  # 构建树树时的采样比率 (0:1]
   # 'min_child_weight':8, # 节点的最少特征数 
    silent= 0,
    learning_rate = 0.1,  # 如同学习率
    seed = 780,
    nthread = 7,
    max_delta_step = 6,
    min_child_weight = 4, #最佳
   # eval_metric = 'logloss',
    n_estimators = 1000)
#modelfit(xgb1,train_data,predictor)
param_test1 = {
 'max_depth':list(range(3,12,2)),
 'min_child_weight':list(range(1,8,2))
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='neg_log_loss',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train_data[predictor],train_data[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
        
        
