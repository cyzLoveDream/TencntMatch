import numpy as np
import pandas as pd
from sklearn.utils import check_random_state 
import time
import sys
import gc
#from joblib import dump, load

sample_pct = 1
tvh = 'N'
xgb_n_trees = 300
nr_bins = int(1e+6)
fre_threshold = 10

base_path = 'D:/MyDeveloperWay/Kaggle/pre/'
data_path = base_path + 'data'
temp_data_path = base_path + 'temp_data'
res_data_path = base_path + 'res_data'
tool_path = base_path + 'tools'
fm_path = tool_path + '/libffm-1.13'
#vw_path = tool_path + '~/vowpal_wabbit/vowpalwabbit/vw '


try:
    params=load(tmp_data_path + '_params.joblib_dat')
    sample_pct = params['pct']
    tvh = params['tvh']
except:
    pass
    
def calcTVTransform(df, vn, vn_y, cred_k, filter_train, mean0=None):
    mean0 = df.ix[filter_train, vn_y].mean()
    print("mean0:%f"%mean0)

    df['_key1'] = df[vn].astype('category').values.codes
    df_yt = df.ix[filter_train, ['_key1', vn_y]]
    #df_y.set_index([')key1'])
    grp1 = df_yt.groupby(['_key1'])
    sum1 = grp1[vn_y].aggregate(np.sum)
    cnt1 = grp1[vn_y].aggregate(np.size)
    v_codes = df.ix[~filter_train, '_key1']
    _sum = sum1[v_codes].values
    _cnt = cnt1[v_codes].values
    _cnt[np.isnan(_sum)] = 0    
    _sum[np.isnan(_sum)] = 0
    r = {}
    r['exp'] = (_sum + cred_k * mean0)/(_cnt + cred_k)
    r['pct'] = _sum /_cnt 
    r['cnt'] = _cnt
    return r
def cntDualKey(df, vn, vn2, key_src, key_tgt, fill_na=False):
    
    print("build src key")
    _key_src = np.add(df[key_src].astype('str').values, df[vn].astype('str').values)
    print ("build tgt key")
    _key_tgt = np.add(df[key_tgt].astype('str').values, df[vn].astype('str').values)
    
    if vn2 is not None:
        _key_src = np.add(_key_src, df[vn2].astype('str').values)
        _key_tgt = np.add(_key_tgt, df[vn2].astype('str').values)

    print( "aggreate by src key")
    grp1 = df.groupby(_key_src)
    cnt1 = grp1[vn].aggregate(np.size)
    
    print ("map to tgt key")
    vn_sum = 'sum_' + vn + '_' + key_src + '_' + key_tgt
    _cnt = cnt1[_key_tgt].values

    if fill_na is not None:
        print ("fill in na")
        _cnt[np.isnan(_cnt)] = fill_na    

    vn_cnt_tgt = 'cnt_' + vn + '_' + key_tgt
    if vn2 is not None:
        vn_cnt_tgt += '_' + vn2
    df[vn_cnt_tgt] = _cnt
    del _key_src
    del _key_tgt
    del grp1
    del cnt1
    del df
    del vn_sum
    del _cnt
    del vn_cnt_tgt
    gc.collect()

def calcDualKey(df, vn, vn2, key_src, key_tgt, vn_y, cred_k, mean0=None, add_count=False, fill_na=False):
    if mean0 is None:
        mean0 = df[vn_y].mean()
    
    print ("build src key")
    _key_src = np.add(df[key_src].astype('str').values, df[vn].astype('str').values)
    print( "build tgt key")
    _key_tgt = np.add(df[key_tgt].astype('str').values, df[vn].astype('str').values)
    
    if vn2 is not None:
        _key_src = np.add(_key_src, df[vn2].astype('str').values)
        _key_tgt = np.add(_key_tgt, df[vn2].astype('str').values)

    print ("aggreate by src key")
    grp1 = df.groupby(_key_src)
    sum1 = grp1[vn_y].aggregate(np.sum)
    cnt1 = grp1[vn_y].aggregate(np.size)
    
    print ("map to tgt key")
    vn_sum = 'sum_' + vn + '_' + key_src + '_' + key_tgt
    _sum = sum1[_key_tgt].values
    _cnt = cnt1[_key_tgt].values

    if fill_na:
        print ("fill in na")
        _cnt[np.isnan(_sum)] = 0    
        _sum[np.isnan(_sum)] = 0

    print ("calc exp")
    if vn2 is not None:
        vn_yexp = 'exp_' + vn + '_' + vn2 + '_' + key_src + '_' + key_tgt
    else:
        vn_yexp = 'exp_' + vn + '_' + key_src + '_' + key_tgt
    df[vn_yexp] = (_sum + cred_k * mean0)/(_cnt + cred_k)

    if add_count:
        print ("add counts")
        vn_cnt_src = 'cnt_' + vn + '_' + key_src
        df[vn_cnt_src] = _cnt
        grp2 = df.groupby(_key_tgt)
        cnt2 = grp2[vn_y].aggregate(np.size)
        _cnt2 = cnt2[_key_tgt].values
        vn_cnt_tgt = 'cnt_' + vn + '_' + key_tgt
        df[vn_cnt_tgt] = _cnt2

def logloss(pred, y, weight=None):
    if weight is None:
        weight = np.ones(y.size)
    
    pred = np.maximum(1e-7, np.minimum(1 - 1e-7, pred))
    return - np.sum(weight * (y * np.log(pred) + (1 - y) * np.log(1 - pred))) / np.sum(weight)
   
def my_grp_cnt(group_by, count_by):
    _ts = time.time()
    _ord = np.lexsort((count_by, group_by))
    print(time.time() - _ts)
    _ts = time.time()    
    _ones = pd.Series(np.ones(group_by.size))
    print(time.time() - _ts)
    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    runnting_cnt = 0
    for i in range(1, group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            if count_by[_ord[i-1]] != count_by[i0]: 
                running_cnt += 1
        else:
            running_cnt = 1
            _prev_grp = group_by[i0]
        if i == group_by.size - 1 or group_by[i0] != group_by[_ord[i+1]]:
            j = i
            while True:
                j0 = _ord[j]
                _cs1[j0] = running_cnt
                if j == 0 or group_by[_ord[j-1]] != group_by[j0]:
                    break
                j -= 1
            
    print(time.time() - _ts)
    if True:
        return _cs1
    else:
        _ts = time.time()    

        org_idx = np.zeros(group_by.size, dtype=np.int)
        print(time.time() - _ts)
        _ts = time.time()    
        org_idx[_ord] = np.asarray(range(group_by.size))
        print(time.time() -_ts)
        _ts = time.time()  
        return _cs1[org_idx]

def calcLeaveOneOut2(df, vn, vn_y, cred_k, r_k, power, mean0=None, add_count=False):
    if mean0 is None:
        mean0 = df_yt[vn_y].mean() * np.ones(df.shape[0])
    _key_codes = df[vn].values.codes
    grp1 = df[vn_y].groupby(_key_codes)
    grp_mean = pd.Series(mean0).groupby(_key_codes) #将mean按照特征进行分组
    mean1 = grp_mean.aggregate(np.mean) #mean分组之后的结果
    sum1 = grp1.aggregate(np.sum)
    cnt1 = grp1.aggregate(np.size)
    
    #print sum1
    #print cnt1
    _sum = sum1[_key_codes].values #新的sun
    _cnt = cnt1[_key_codes].values
    _mean = mean1[_key_codes].values
    #print _sum[:10]
    #print _cnt[:10]
    #print _mean[:10]
    #print _cnt[:10]
    _mean[np.isnan(_sum)] = mean0.mean()
    _cnt[np.isnan(_sum)] = 0    
    _sum[np.isnan(_sum)] = 0
    #print _cnt[:10]
    _sum -= df[vn_y].values
    _cnt -= 1
    #print _cnt[:10]
    vn_yexp = 'exp2_'+vn
#    df[vn_yexp] = (_sum + cred_k * mean0)/(_cnt + cred_k)
    diff = np.power((_sum + cred_k * _mean)/(_cnt + cred_k) / _mean, power)
    if vn_yexp in df.columns:
        df[vn_yexp] *= diff
    else:
        df[vn_yexp] = diff 
    if r_k > 0:
        df[vn_yexp] *= np.exp((np.random.rand(np.sum(filter_train))-.5) * r_k)
    if add_count:
        df[vn_cnt] = _cnt
    return diff

def mergeLeaveOneOut2(df, dfv, vn):
    _key_codes = df[vn].values.codes
    vn_yexp = 'exp2_'+vn
    grp1 = df[vn_yexp].groupby(_key_codes)
    _mean1 = grp1.aggregate(np.mean)
    
    _mean = _mean1[dfv[vn].values.codes].values
    
    _mean[np.isnan(_mean)] = _mean1.mean()

    return _mean

def my_grp_idx(group_by, order_by):
    _ts = time.time()
    _ord = np.lexsort((order_by, group_by))
    print(time.time() - _ts)
    _ts = time.time()    
    _ones = pd.Series(np.ones(group_by.size))
    print(time.time() - _ts)
    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    for i in range(1, group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            _cs1[i] = _cs1[i - 1] + 1
        else:
            _cs1[i] = 1
            _prev_grp = group_by[i0]
    print(time.time() - _ts)
    _ts = time.time()    
    
    org_idx = np.zeros(group_by.size, dtype=np.int)
    print(time.time() - _ts)
    _ts = time.time()    
    org_idx[_ord] = np.asarray(range(group_by.size))
    print(time.time() - _ts)
    _ts = time.time()    

    return _cs1[org_idx]

def get_agg(group_by, value, func):
    g1 = pd.Series(value).groupby(group_by)
    agg1  = g1.aggregate(func)
    #print agg1
    r1 = agg1[group_by].values
    return r1
   
def calc_exptv(t0, vn_list, last_day_only=False, add_count=True):
    t0a = t0.copy()
    day_exps = {}

    for vn in vn_list:
        t0a[vn] = t0[vn]
        for day_v in range(18, 31):
            cred_k = 10
            if day_v not in day_exps:
                day_exps[day_v] = {}

            vn_key = vn

            import time
            _tstart = time.time()

            day1 = 16
            if last_day_only:
                day1 = day_v - 2
            #今天以及今天之前的数据
            filter_t = np.logical_and(t0.day.values > day1, t0.day.values <= day_v)
            vn_key = vn
            #不包括今天的数据
            t1 = t0a.ix[filter_t, :].copy()
            filter_t2 = np.logical_and(t1.day.values != day_v, t1.day.values < 31)
            
            if last_day_only:
                day_exps[day_v][vn_key] = calcTVTransform(t1, vn, 'label', cred_k, filter_t2)
            else:
                day_exps[day_v][vn_key] = calcTVTransform(t1, vn, 'label', cred_k, filter_t2)
            
            print(vn, vn_key, " ", day_v, " done in ", time.time() - _tstart)
        t0a.drop(vn, inplace=True, axis=1)
        
    for vn in vn_list:
        vn_key = vn
            
        vn_exp = 'exptv_'+vn_key
        if last_day_only:
            vn_exp='expld_'+vn_key
            
        t0[vn_exp] = np.zeros(t0.shape[0])
        if add_count:
            t0['cnttv_'+vn_key] = np.zeros(t0.shape[0])
        for day_v in range(18, 31):
            print(vn, vn_key, day_v, t0.ix[t0.day.values == day_v, vn_exp].values.size, day_exps[day_v][vn_key]['exp'].size)
            t0.loc[t0.day.values == day_v, vn_exp]=day_exps[day_v][vn_key]['exp']
            if add_count:
                t0.loc[t0.day.values == day_v, 'cnttv_'+vn_key]=day_exps[day_v][vn_key]['cnt']
        
