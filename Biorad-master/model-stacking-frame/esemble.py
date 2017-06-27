import time
import pandas as pd
from math import exp, log, sqrt
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np

## import xgboost as xgb
import lightgbm as lgb
import utils
from utils import *
start_time = time.time()


def logloss(pred, y, weight=None):
    if weight is None:
        weight = np.ones(y.size)
    
    pred = np.maximum(1e-7, np.minimum(1 - 1e-7, pred))
    return - np.sum(weight * (y * np.log(pred) + (1 - y) * np.log(1 - pred))) / np.sum(weight)


class Ensemble(object):
    def __init__(self, n_folds, base_models):
        self.n_folds = n_folds
        self.base_models = base_models
    
    
    def fit_pred(self, df_all):
        df_train = df_all[df_all.day < 30]
        df_valid = df_all[df_all.day == 30]
        df_test = df_all[df_all.day == 31]
        
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=2017)
        s_train = np.zeros((df_train.shape[0], len(self.base_models)))
        s_valid = np.zeros((df_valid.shape[0], len(self.base_models)))
        s_test = np.zeros((df_test.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            s_valid_i = np.zeros((df_valid.shape[0], self.n_folds))
            s_test_i = np.zeros((df_test.shape[0], self.n_folds))
            j = 0
            for train_idx, holdout_idx in folds.split(df_train):
                train = df_train.iloc[train_idx]
                holdout = df_train.iloc[holdout_idx]
                clf.fit(train, df_valid)
                s_train[holdout_idx, i]=clf.predict(holdout)[:]
                s_valid_i[:, j]=clf.predict(df_valid)[:]
                s_test_i[:, j]=clf.predict(df_test)[:]
                print('%d folds Elapsed: %s minutes ---'%(j, round(((time.time()-start_time)/60), 2)))
                j += 1
            s_valid[:, i] = s_valid_i.mean(1)
            s_test[:, i] = s_test_i.mean(1)
            print('%d base_model Elapsed: %s minutes ---'%(i, round(((time.time()-start_time)/60), 2)))
        
        lr = LogisticRegression()
        lr.fit(s_train, df_train['label'])
        proba_train = lr.predict_proba(s_train)[:, 1]
        loss_train = logloss(proba_train, df_train['label'])
        print(' train logloss: ', loss_train)
        
        proba_valid = lr.predict_proba(s_valid)[:, 1]
        loss_valid = logloss(proba_valid, df_valid['label'])
        print(' valid logloss: ', loss_valid)
        
        proba_test = lr.predict_proba(s_test)[:, 1]
        return proba_test


class LGB(object):
    def __init__(self, params,num_boost_round, fea_list):
        super(LGB, self).__init__()
        self.params = params
        self.num_boost_round = num_boost_round
        self.fea_list = fea_list
    
    def fit(self, train, valid):
        lgb_train = lgb.Dataset(train[self.fea_list], train['label'])
        lgb_val = lgb.Dataset(valid[self.fea_list], valid['label'])
        
        self.gbm = lgb.train(self.params, lgb_train, self.num_boost_round, valid_sets=lgb_val,
                             feature_name=self.fea_list, early_stopping_rounds=20)
    
    def predict(self, x):
        y_pred = self.gbm.predict(x[self.fea_list])
        return y_pred

class XGB(object):
    def __init__(self, param_xgb, num_boost_round, fea_list):
        super(XGB, self).__init__()
        self.params = param_xgb
        self.num_boost_round = num_boost_round
        self.fea_list = fea_list
    
    def fit(self, train, valid):
        xgb_train = xgb.DMatrix(train[self.fea_list], train['label'])
        xgb_val = xgb.DMatrix(valid[self.fea_list], valid['label'])
        
        watchlist = [(xgb_train, 'train'), (xgb_val, 'valid')]
        
        plst = list(params.items()) + [('eval_metric', 'logloss')]
        self.xgb = xgb.train(plst, train, xgb_n_trees, watchlist)
    
    def predict(self, x):
        y_pred = self.xgb.predict(x[self.fea_list])
        return y_pred

class FTRL(object):
    def __init__(self, alpha, beta, L1, L2, D, epoch, fea_list):
        self.epoch = epoch
        self.D = D
        self.learner = ftrl_proximal(alpha, beta, L1, L2, D)
        self.fea_list = fea_list
    
    def gen_data(self, data):
        
        for t, row in data.iterrows():
            # process id
            ID = int(row['instanceID'])
            # print(t,row)
            del row['instanceID']
            
            # process clicks
            y = 0.
            if row['label'] == '1':
                y = 1.
            del row['label']
            date = int(float(row['day']))
            
            x = []
            for key in self.fea_list:
                value = row[key]
                # one-hot encode everything with hash trick
                index = abs(hash(key + '_' + str(value))) % self.D
                x.append(index)
            yield t, date, ID, x, y
    def logloss(self, p, y):
        
        p = max(min(p, 1. - 10e-15), 10e-15)
        return -log(p) if y == 1. else -log(1. - p)
    def fit(self, train, valid):
        frame = [train, valid]
        data = pd.concat(frame)
        for e in range(self.epoch):
            loss = 0
            count = 0
            instanceID = 0
            test_sum = 0
            for t, date, ID, x, y in self.gen_data(data):
                p = self.learner.predict(x)
                if date == 30:
                    loss += self.logloss(p, y)
                    if count % 100000 == 0:
                        print(count)
                    count += 1
                else:
                    if ID % 1000000 == 0:
                        print(str(time.time() - start_time), ID)
                    self.learner.update(x, p, y)
            
            print('Epoch %d finished, validation logloss: %f, elapsed time: %s' % (
                e, loss/count, str(time.time() - start_time)))
    
    def predict(self, test):
        res = []
        for t, date, ID, x, y in self.gen_data(test):
            p = self.learner.predict(x)
            res.append(p)
        return res


class ftrl_proximal(object):
    
    def __init__(self, alpha, beta, L1, L2, D):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2
        
        # feature related parameters
        self.D = D
        self.interaction = False
        
        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}
    
    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''
        
        # first yield index of the bias term
        yield 0
        
        # then yield the normal indices
        for index in x:
            yield index
        
        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)
            
            x = sorted(x)
            for i in range(L):
                for j in range(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D
    
    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''
        
        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2
        
        # model
        n = self.n
        z = self.z
        w = {}
        
        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]
            
            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)
            
            wTx += w[i]
        
        # cache the current w for update stage
        self.w = w
        
        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))
    
    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''
        
        # parameter
        alpha = self.alpha
        
        # model
        n = self.n
        z = self.z
        w = self.w
        
        # gradient under logloss
        g = p - y
        
        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g

def main(input = '../test/data_test.csv'):
    data = pd.read_csv(input)
    
    fea_list = [ 'connectionType', 'creativeID',
                 'positionID', 'telecomsOperator', 'userID', 'day', 'adID',
                 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'sitesetID',
                 'positionType', 'age', 'gender', 'education', 'marriageStatus',
                 'haveBaby', 'hometown', 'residence', 'appCategory']
    
    print('--- Features Set: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Number of Features: ', len(data.columns.tolist()))
    
    
    params_lgb = {'boosting_type': 'gbdt','objective': 'binary','metric': 'binary_logloss',
                  'num_leaves': 31,'learning_rate': 0.05,'feature_fraction': 0.9,'bagging_fraction': 0.8,
                  'bagging_freq': 5,'verbose': 0}
    
    param_xgb = {'max_depth':15, 'eta':.02, 'objective':'binary:logistic', 'verbose':0,
                 'subsample':1.0, 'min_child_weight':50, 'gamma':0,
                 'nthread': 16, 'colsample_bytree':.5, 'base_score':0.16, 'seed': 999}
    
    # params for ftrl
    alpha = .1  # learning rate
    beta = 1.   # smoothing parameter for adaptive learning rate
    L1 = 1.     # L1 regularization, larger value means more regularized
    L2 = 1.     # L2 regularization, larger value means more regularized
    epoch = 3
    D = 2**20
    
    # base_models = [    
    #     LGB(params_lgb, 200, FIELDS_0),
    #     XGB(param_xgb, 1000, FIELDS_0),
    #     FTRL(alpha, beta, L1, L2, D, epoch)
    # ]
    
    base_models = [
        FTRL(alpha, beta, L1, L2, D, epoch, fea_list),
        LGB(params_lgb, 200, fea_list),
        LGB(params_lgb, 200, fea_list),
        ]
    
    ensemble = Ensemble(n_folds=2, base_models=base_models)
    y_pred = ensemble.fit_pred(data)
    
    df = pd.DataFrame({"instanceID": range(1, len(y_pred) + 1), "proba": y_pred})
    df.sort_values("instanceID", inplace=True)
    df.to_csv("/submission.csv", index=False)
    print('--- Submission Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

if __name__ == '__main__':
    main()






