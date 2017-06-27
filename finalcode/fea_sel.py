# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 22:36:04 2017

@author: Administrator
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from minepy import MINE
import pandas as pd
import time
import gc
now = time.time()
#读取数据
train = pd.read_pickle(r'E:\finaldata\final\sample\feadata\train')
features = [x for x in train.columns if x not in ['label','instanceID','day','clickTime','userID','conversionTime']]
print("data input finish ")
#各种方法进行特征选择
names = features
X = np.array(train[features].values)
Y = np.array(train['label'].values)
ranks = {}
def rank_to_dict(ranks, names, order=1):
	minmax = MinMaxScaler()
	ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
	ranks = map(lambda x: round(x, 2), ranks)
	return dict(zip(names, ranks ))
lr = LinearRegression(normalize=True)
lr.fit(X, Y)
ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)
del lr
gc.collect()
print("lr is finish in time {0}".format(time.time()-now))
now =time.time()
ridge = Ridge(alpha=7)
ridge.fit(X, Y)
ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)
del ridge
gc.collect()
print("ridge is finish in time {0}".format(time.time()-now))
now = time.time()
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)
del lasso
gc.collect()
print("lasso is finish in time {0}".format(time.time()-now)) 
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)
del rlasso
gc.collect()
print("stability is finish in time {0}".format(time.time()-now))

#stop the search when 5 features are left (they will get equal scores)
rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X,Y)
ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), names, order=-1)
del rfe
gc.collect()
print("ref is finish in time {0}".format(time.time()-now))
rf = RandomForestRegressor()
rf.fit(X,Y)
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)
del rf
gc.collect()
print("rf is finish in time {0}".format(time.time()-now))
f, pval  = f_regression(X, Y, center=True)
ranks["Corr."] = rank_to_dict(f, names)
mine = MINE()
mic_scores = []
for i in range(X.shape[1]):
	mine.compute_score(X[:,i], Y)
	m = mine.mic()
	mic_scores.append(m)
ranks["MIC"] = rank_to_dict(mic_scores, names)
r = {}
for name in names:
	r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
print ("\t%s" % "\t".join(methods))
for name in names:
	print ("%s\t%s" % (name, "\t".join(map(str, [ranks[method][name] for method in methods]))))