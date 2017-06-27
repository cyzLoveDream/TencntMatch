# -*- coding: utf-8 -*-
"""
Created on Sat May 13 16:34:24 2017

@author: Administrator
"""

from sklearn.model_selection import train_test_split
from pandas import DataFrame
from sklearn import metrics
from sklearn.datasets import make_hastie_10_2
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_importance
import matplotlib.pyplot as plt
from graphviz import Digraph

x,y = make_hastie_10_2(random_state=0)
x = DataFrame(x)
y = DataFrame(y)
y.columns = {'label'}
label = {-1:0,1:1}
y.label = y.label.map(label)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)#划分数据集
#XGBoost自带接口
params={
    'eta': 0.3,
    'max_depth':3,   
    'min_child_weight':1,
    'gamma':0.3, 
    'subsample':0.8,
    'colsample_bytree':0.8,
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'nthread':12,
    'scale_pos_weight': 1,
    'lambda':1,  
    'seed':27,
    'silent':0 ,
    'eval_metric': 'auc'
}
d_train = xgb.DMatrix(X_train,label=y_train)
d_vaild = xgb.DMatrix(X_test,label=y_test)
d_test = xgb.DMatrix(X_test)
watchlist = [(d_train,'dtrain'),(d_vaild,'dvaild')]
#sklearn接口
clf = XGBClassifier(
    n_estimators=30,#三十棵树
    learning_rate =0.3,
    max_depth=3,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=27)
model_bst = xgb.train(params, d_train, 30, watchlist, early_stopping_rounds=500, verbose_eval=10)
model_sklearn=clf.fit(X_train, y_train) 
y_bst= model_bst.predict(d_test)
y_sklearn= clf.predict_proba(X_test)[:,1]
print("XGBoost_自带接口    AUC Score : %f" % metrics.roc_auc_score(y_test, y_bst))
print("XGBoost_sklearn接口 AUC Score : %f" % metrics.roc_auc_score(y_test, y_sklearn))
print('原始train大小',X_train.shape)
print('原始test大小',X_test.shape)
##XGBoost自带接口生成的新特征
train_new_feature = model_bst.predict(d_train,pred_leaf=True)
test_new_feature = model_bst.predict(d_test,pred_leaf=True)
train_new_feature = DataFrame(train_new_feature)
test_new_feature = DataFrame(test_new_feature)
print('生成train特征大小',train_new_feature.shape)
print('生成test特征大小',test_new_feature.shape)
#sklearn接口生成的新特征
train_sk_new_feature = clf.apply(X_train)
test_sk_new_feature = clf.apply(X_test)
train_sk_new_feature = DataFrame(train_sk_new_feature)
test_sk_new_feature = DataFrame(test_sk_new_feature)
print('sk生成train特征大小',train_sk_new_feature.shape)
print('sk生成test特征大小',test_sk_new_feature.shape)
#用XGBoost自带接口生成的新特征训练
new_feature1=clf.fit(train_new_feature, y_train)
y_new_feature1= clf.predict_proba(test_new_feature)[:,1]
#用XGBoost自带接口生成的新特征训练
new_feature2=clf.fit(train_sk_new_feature, y_train)
y_new_feature2= clf.predict_proba(test_sk_new_feature)[:,1]
print("XGBoost自带接口生成的新特征预测结果 AUC Score : %f" % metrics.roc_auc_score(y_test, y_new_feature1))
print("XGBoost自带接口生成的新特征预测结果 AUC Score : %f" % metrics.roc_auc_score(y_test, y_new_feature2))
plot_tree(model_bst,num_trees=1)
plot_importance(model_bst)
plt.show()