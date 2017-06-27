

import time
import gc
import numpy as np
import pandas as pd 
'''
  df :DataFrame 需要排序的结构
  pre_num_rank : 需要排序的变量 例如用户的全局统计变量cnt_userID
'''
def get_total_num_rank_fea(df,pre_num_rank='cnt_userID',creative = 'creativeID',every=False,other_key = 'userID'): 
   
    now = time.time()
    print("produce dict from df in {0:6.0f}".format(time.time()-now))  
    if every == True:
        pcdit = {}
        #用户的唯一值 
        cucu = df[[other_key,creative,pre_num_rank]].sort_values(by=other_key)
            #df中用户排序
        user = cucu[other_key]
        #df中用户排序后的cnt值
        cuid = cucu[pre_num_rank]
        #构建用户字典形如{1:[1,2,3],2:[1,4,2]}
        inde = list(cucu.index)
        print("begin produce user dict")
        now = time.time()
        i=0
        for index in inde:
            i += 1
            if i % 500000 ==0:
                print("have been handle user is {0}".format(i))  
            s = [cuid[index]]
            if user[index] not in pcdit:
                pcdit.setdefault(user[index],s)
            else:
                m = cuid[index]
                pcdit.get(user[index]).append(m) 
        print("finish in time {0:6.0f}".format(time.time()-now))
        #将用户字典排序 
        #利用排序后的用户字典构建新的用户字典
        now = time.time() 
        print("begin produce feature")
        for pcdit_key in pcdit.keys():
            pcdit.get(pcdit_key).sort()
            #ps是排序后用户的点击数字典 形如 {1:{1:0,2:1,3:2},2:{1:0},3:{1:0}}
            ps = pcdit.get(pcdit_key)
            psdic = {}
            for i in ps:
                if i not in psdic:
                    psdic.setdefault(i,len(psdic))
            pcdit[pcdit_key] = psdic
        print("finish in time {0:6.0f}".format(time.time()-now))  
        _eur = pd.Series(np.zeros(cucu.shape[0]),index=cucu.index)
        #给DF中赋值
        for s in inde:
            _eur[s] = pcdit.get(user[s]).get(cuid[s])
        print("finish")
        return _eur
    else:
        #只是构建一个字典用来查询  全局字典
        cnt_dic = {}
        pres = np.array(df[pre_num_rank])
        pres.sort()
        for index in pres: 
            if pres[index] not in cnt_dic:
                cnt_dic.setdefault(pres[index],len(cnt_dic))  
        print("finish get dict in time {0:6.0f}".format(time.time()-now)) 
        _pnr = pd.Series(np.zeros(pres.size,dtype=np.int),index = df.index)
        inde = list(index)
        #出现问题，正在修改(已改)
        for line in inde:
            if line % 500000 ==0:
                print("have been handle size is {0}".format(line))
            _pnr[line] = cnt_dic.get(df.ix[line,pre_num_rank])
        now = time.time()
        print("produce rank fea in time {0:6.0f}".format(time.time()-now))
        return _pnr

'''
    每天的用户点击creative前后顺序按照时间排序
'''

def get_preOrNextClickRank_in_day(df,clickTime = 'clickTime',creative='creativeID',user = 'userID',day_begin = 17,day_end = 32): 
    _preOrNextClickRank = pd.Series(np.zeros(df.shape[0]),index = df.index)
    for day in range(day_begin,day_end):
        now1 = time.time() 
        df_day = df[df['day']==day]
        pcdit = {}
        #用户的唯一值 
        cucu = df_day[[user,creative,clickTime]].sort_values(by=user)
            #df中用户排序
        us = cucu[user]
        #df中用户排序后的cnt值
        ct = cucu[clickTime]
        cre = cucu[creative]
        inde = list(cucu.index) 
        #构建用户字典形如{(user,creative):[clickTime1,clickTime2]}
        print("begin produce user dict")
        now = time.time()
        i=0
        for index in inde: 
            i += 1
            if i % 500000 ==0:
                print("have been handle user is {0}".format(i))  
            cu =(us[index],cre[index])
            s = [ct[index]]
            if cu not in pcdit:
                pcdit.setdefault(cu,s)
            else:
                m = ct[index]
                pcdit.get(cu).append(m)  
        print("finish in time {0:6.0f}".format(time.time()-now))
        #将用户字典排序 
        #利用排序后的用户字典构建新的用户字典
        now = time.time() 
        print("begin produce feature")
        for pcdit_key in pcdit.keys():
            pcdit.get(pcdit_key).sort()
            #ps是排序后用户的点击数字典 形如 {(user,creative):{clickTime1:0,clickTime2:1,clickTime3:2}}
            ps = pcdit.get(pcdit_key)
            psdic = {}
            for i in ps:
                if i not in psdic:
                    psdic.setdefault(i,len(psdic))
            pcdit[pcdit_key] = psdic
        print("finish in time {0:6.0f}".format(time.time()-now))  
        _eur = pd.Series(np.zeros(cucu.shape[0]),index=cucu.index) 
        #代码问题，正在修改(已改好)
        i=0
        for s in inde:
            i += 1
            if i % 500000 ==0:
                print("have been handle user is {0}".format(i)) 
            u = (us[s],cre[s])
            _eur[s] = pcdit.get(u).get(ct[s])
        _preOrNextClickRank.update(_eur)
        print("finish the day {0} in time {1}".format(day,time.time()-now1))
    return _preOrNextClickRank
        
    '''
    按照每天进行，对用户的点击次数排序,并将排序后的index作为特征返回，且返回一个Series
    '''
def get_cntRank_sameUserIDClickCnt_in_day(df,pre_click_num='cnt_userID',creative = 'creativeID',otherkey = 'userID',day_begin=17,day_end=32):
    _totalrsuicc = pd.Series(np.zeros(df.shape[0]),index=df.index) 
    for day in range(day_begin,day_end):
        now =time.time()
        df_day = df[df['day']==day]
        print("begin handle the day {0}".format(day))
        _day_rsuicc = get_total_num_rank_fea(df_day,pre_click_num,creative,every=True,other_key=otherkey)
        print("finish handle this day")
        del df_day
        #更新到_totalrsuicc中
        _totalrsuicc.update(_day_rsuicc)
        print("finish handle the day {0} in time {1}".format(day,time.time()-now))
    return _totalrsuicc
   
'''
    按照每天对用户的转化次数进行排序，并将排序后的index作为特征返回，且返回一个Series
'''
def get_covRank_sameUserIDClickCnt_in_day(df,pre_cov_cnt='cov_cnt_userID',creative='creativeID',day_begin=17,day_end=32,ok='userID'):
    _totalCovRank = pd.Series(np.zeros(df.shape[0]),index = df.index)
    now1 = time.time()
    for day in range(day_begin,day_end):
        now = time.time()
        df_day = df[df['day']==day]
        print("begin handle the day {0}".format(day))
        _day_covRank = get_total_num_rank_fea(df_day,pre_cov_cnt,creative,every=True,other_key=ok)
        print("--------------------finish get_total_num_rank_fea-------------------------------------------------------------------")
        print("finish handle this day")
        _totalCovRank.update(_day_covRank)
        del df_day
        print("finish handle the day {0} in time {1}".format(day,time.time()-now))
    print("finish handle the day {0} in time {1}".format(day,time.time()-now1))
    return _totalCovRank

'''
    统计每天的转化率以及转化率排序特征，并返回两个值，第一个是每天的转化率特征，第二个值是转化率排序特征
'''
def get_covRatioAndRank_sameUserIDCreativeID_in_day(df,uiD='userID',creative ='creativeID',label='label',day_begin=17,day_end=32):
    _covRatio = pd.Series(np.zeros(df.shape[0]),index = df.index)
    _covRatioRank = pd.Series(np.zeros(df.shape[0]),index = df.index)
    new_cnt ='cnt_' + uiD +'_' + creative
    new_cov = 'cov_' +uiD + '_' +creative
    df[new_cnt],df[new_cov] = get_cntAndCovCnt_every_day(df,uiD,creative,label,day_begin,day_end)
    print("--------------------finish get_cntAndCovCnt_every_day-------------------------------------------------------------------")
    _covRatio = df[new_cov] / df[new_cnt]
    ratio = 'covRatio_' + uiD + '_' +creative
    df[ratio] = _covRatio
    _covRatioRank = get_covRank_sameUserIDClickCnt_in_day(df,ratio,creative,day_begin,day_end,ok=uiD)
    print("--------------------finish get_covRank_sameUserIDClickCnt_in_day-------------------------------------------------------------------")
    return _covRatio,_covRatioRank
'''
    统计每一个用户在不同的creativeID下的点击次数（全局）
    返回一个新的列
'''
def get_user_ever_num(df,uiD='userID',cid = 'creativeID'):
    ucid = {}
    now = time.time()
    print("begin produce dict") 
    user =df[uiD]
    cre = df[cid]
    inde =list(user.index)
    i=0
    for line in inde:
        i += 1
        if i % 500000 ==0:
            print("have been handle size is {0}".format(i)) 
        uc = (user[line],cre[line])
        if uc not in ucid:
            ucid.setdefault(uc,1)
        else:
            ucid[uc] += 1  
    print("finish produce dict in time {0:6.0f}".format(time.time()-now))
    print("produce feature")
    now = time.time() 
    _cs1 =pd.Series(np.zeros(df.shape[0],dtype=np.int),index = df.index)
    i=0
    for line in inde:
        i += 1
        if i % 500000 ==0:
            print("have been handle size is {0}".format(i))
        uc = (user[line],cre[line]) 
        _cs1[line] = ucid.get(uc) 
    print("finish produce feature in time {0:6.0f}".format(time.time()-now))
    return _cs1

'''
    获取用户每天的转化次数以及点击次数,返回两个值，第一个值是点击次数，第二个值是转化次数
'''
def get_cntAndCovCnt_every_day(df,userID='userID',creative ='creativeID',label='label',day_begin=17,day_end=32):
    _cnt = pd.Series(np.zeros(df.shape[0]),index = df.index)
    _covCnt = pd.Series(np.zeros(df.shape[0]),index = df.index) 
    print("begin product dict")
    for day in range(day_begin,day_end): 
        now1 = time.time() 
        now = now1
        df_day = df[df['day']==day]
        user = df_day[userID]
        cre = df_day[creative]
        l = df_day[label]
        cnt = {}
        cov = {}
        inde = list(df_day.index)
        i = 0
        for line in inde:
            i += 1
            if i % 500000 ==0:
                print("have been handle size is {0}".format(i))
            uc = (user[line],cre[line])
            if uc not in cnt:
                cnt.setdefault(uc,1)
            else:
                cnt[uc] += 1  
            if l[line]==1:
                if uc not in cov:
                    cov.setdefault(uc,1)
                else:
                    cov[uc] +=1
        print("finish produce dict in time {0:6.0f}".format(time.time()-now))
        print("produce feature")
        now = time.time() 
        _cnt_day = pd.Series(np.zeros(df_day.shape[0],dtype=np.int),index = df_day.index)
        _covCnt_day = pd.Series(np.zeros(df_day.shape[0],dtype=np.int),index = df_day.index)
        i=0
        for line in inde: 
            i += 1
            if i % 500000 ==0:
                print("have been handle size is {0}".format(i))
            uc = (user[line],cre[line])   
            _cnt_day[line] = cnt.get(uc)
            _covCnt_day[line] = cov.get(uc)
        print("finish produce feature in time {0:6.0f}".format(time.time()-now))
        _cnt.update(_cnt_day)
        _covCnt.update(_covCnt_day)
        print("finish handle the day {0} in time {1}".format(day,time.time()-now1))
        del _cnt_day
        del _covCnt_day
        del df_day 
        del cre
        del user 
        del l
        gc.collect()
    return _cnt,_covCnt

'''
    产生捆绑APP的特征，定义一个目录下APPID小于等于2个的为捆绑APP
'''
def get_bindApp_fea(df,act = 'appCategory',app='appID',size = 3):
    #建立一个{目录：【APPID】}的字典 
    a = df[act]
    p = df[app] 
    app_dict = {}
    inde = list(df.index)
    print("begin product dict")
    now1 = time.time()
    now = now1
    s = 0
    for i in inde: 
        s += 1
        if s % 500000 ==0:
             print("have been handle size is {0}".format(s))
        cate = a[i]
        ap = [p[i]]
        if cate not in app_dict:
            app_dict.setdefault(cate,ap)
        else:
            ap1 = p[i]
            if ap1 not in app_dict.get(cate):
                app_dict.get(cate).append(ap1)
    print("finish produce dict in time {0}".format(time.time()-now))
    #生成特征
    _bindApp_fea = pd.Series(np.zeros(df.shape[0]),index = df.index)
    s = 0 
    for i in inde:
        s += 1
        if s % 500000 ==0:
             print("have been handle size is {0}".format(s))
        # 可以定义长度小于几个的默认捆绑
        if len(app_dict.get(a[i])) < size:
            _bindApp_fea[i] = 1
    print("finsih product feature")
    print("finish produce this feauture in time {0:6.0f}".format(time.time()-now1))
    return _bindApp_fea
def get_cnt_single(df,group,lab = 'label',day_begin = 28,day_end = 32):
    grp_dict = {}
    grp_cov_dict = {}
    data_df = df[['label',group]].sort_values(by=group)
    grp = data_df[group]
    lab = data_df['label']
    index = list(data_df.index)
    _cnt_group = pd.Series(np.zeros(df.shape[0]),index = df.index)
    _cnt_cov_group =pd.Series(np.zeros(df.shape[0]),index = df.index)
    now = time.time()
    #构建整体字典
    s = 0
    print("begin produce history dict")
    for inde in index:
        s += 1
        if s % 500000 ==0:
             print("have been handle size is {0}".format(s))
        g = grp[inde]
        l = lab[inde]
        if g not in grp_dict:
            grp_dict.setdefault(g,1)
        else:
            grp_dict[g] +=1
        if l == 1:
            if g not in grp_cov_dict:
                grp_cov_dict.setdefault(g,1)
            else:
                grp_cov_dict[g] += 1
    #产生当天的统计特征
    for day in range(day_begin,day_end):
        day_grp_dict = {}
        day_grp_cov_dict = {}
        day_data = df[df['day']==day]
        g_d = day_data[group]
        l_d = day_data['label']
        index_day = list(day_data.index)
        _cnt_group_day = pd.Series(np.zeros(day_data.shape[0]),index = day_data.index)
        _cnt_cov_group_day = pd.Series(np.zeros(day_data.shape[0]),index = day_data.index)
        s = 0
        print("begin produce history dict")
        for inde_day in index_day:
            s += 1
            if s % 500000 ==0:
                 print("have been handle size is {0}".format(s))
            g = g_d[inde_day]
            l = l_d[inde_day]
            if g not in day_grp_dict:
                day_grp_dict.setdefault(g,1)
            else:
                day_grp_dict[g] +=1
            if l == 1:
                if g not in day_grp_cov_dict:
                    day_grp_cov_dict.setdefault(g,1)
                else:
                    day_grp_cov_dict[g] += 1
        print("finish produce history")
        print("produce feature")
        s = 0
        for i in index_day:
            s += 1
            if s % 500000 ==0:
                 print("have been handle size is {0}".format(s))
            g1 = g_d[i]
            try:
                g_d_g = grp_dict.get(g1)
            except:
                g_d_g = 0
            try:
                d_g_d_g = day_grp_dict.get(g1)
            except:
                d_g_d_g = 0 
            if d_g_d_g ==None:
                d_g_d_g = 0
            if g_d_g ==None:
                g_d_g = 0
            cha =int(g_d_g) - int(d_g_d_g)
            try:
                g_d_c = grp_cov_dict.get(g1)
            except:
                g_d_c = 0
            try:
                d_g_c_g = day_grp_cov_dict.get(g1)
            except:
                d_g_c_g = 0 
            if d_g_c_g == None:
                d_g_c_g = 0
            if g_d_c == None:
                g_d_c = 0
            cov_cha = int(g_d_c) - int(d_g_c_g)
            _cnt_group_day[i] = cha
            _cnt_cov_group_day[i] = cov_cha
        _cnt_group.update(_cnt_group_day)
        _cnt_cov_group.update(_cnt_cov_group_day)
    return _cnt_group,_cnt_cov_group

def get_staCnt_from_combine(df,df_his,lab = 'label',cre='creativeID',groupby=None,day_begin = 28,day_end = 32):
    # 构建历史字典
    cre_dict = {}
    cre_cov_dict = {}
    data_his = df_his[[cre,groupby,'clickTime',lab]].sort_values(by = [cre,'clickTime'])
    lab_his = data_his[lab]
    cre_his = data_his[cre]
    grp_his = data_his[groupby]
    index_his = list(data_his.index)
    now = time.time()
    s = 0
    print("begin produce history dict")
    for inde_his in index_his:
        s += 1
        if s % 500000 ==0:
             print("have been handle size is {0}".format(s))
        c_h = cre_his[inde_his]
        g_h = grp_his[inde_his]
        l_h = lab_his[inde_his]
        c_g_h = (c_h,g_h)
        if c_g_h not in cre_dict:
            cre_dict.setdefault(c_g_h,1)
        else:
            cre_dict[c_g_h] += 1
        if l_h == 1:
            if c_g_h not in cre_cov_dict:
                cre_cov_dict.setdefault(c_g_h,1)
            else:
                cre_cov_dict[c_g_h] += 1 
    print("finish produce history dict in time {0:6.0f}".format(time.time()-now))
    #完成构建字典
    tra_data = df[[cre,groupby,'clickTime',lab]].sort_values(by = [cre,'clickTime'])
    tra_cre = df[cre]
    tra_grp = df[groupby]
    tra_lab = df[lab]
    tra_index = list(tra_data.index)
    _cnt_cre_group = pd.Series(np.zeros(tra_data.shape[0]),index = tra_data.index)
    _cnt_cov_cre_group = pd.Series(np.zeros(tra_data.shape[0]),index = tra_data.index)
    for day in range(day_begin,day_end):
        #处理数据,先赋值再修改字典
        df_day = df[df['day']==day]
        _cnt_cre_day = pd.Series(np.zeros(df_day.shape[0]),index = df_day.index)
        _cov_cre_day = pd.Series(np.zeros(df_day.shape[0]),index = df_day.index) 
        df_day_data = df_day[[cre,groupby,lab]]
        df_day_cre = df_day_data[cre]
        df_day_grp = df_day_data[groupby]
        df_day_lab = df_day_data[lab]
        index_day = list(df_day_data.index)
        s = 0
        for inde_day in index_day:
            s += 1
            if s % 500000 ==0:
                 print("have been handle size is {0}".format(s))
            c_d = df_day_cre[inde_day]
            g_d = df_day_grp[inde_day]
            l_d = df_day_lab[inde_day]
            c_g_d = (c_d,g_d)
            _cnt_cre_day[inde_day] = cre_dict.get(c_g_d)
            _cov_cre_day[inde_day] = cre_cov_dict.get(c_g_d)
            if c_g_d not in cre_dict:
                cre_dict.setdefault(c_g_d,1)
            else:
                cre_dict[c_g_d] += 1
            if l_d == 1:
                if c_g_d not in cre_cov_dict:
                    cre_cov_dict.setdefault(c_g_d,1)
                else:
                    cre_cov_dict[c_g_d] += 1
        _cnt_cre_group.update(_cnt_cre_day)
        _cnt_cov_cre_group.update(_cov_cre_day)
        print("finish the day {0}".format(day))
    return _cnt_cre_group,_cnt_cov_cre_group 

'''
    处理app的字符串
'''
def handle_app_str(s): 
    s = s[1:len(s)-1]
    s = s.split(',')
    if len(s) == 1:
        s = []
    else:
       for i in range(0,len(s)):
            s[i] = int(s[i])
    return s
'''
    恢复字典形式
'''
def get_user_dict(df_dict):
    re_dict = {}
    user = df_dict['userID']
    app = df_dict['0']
    index = list(df_dict.index)
    s = 0
    for inde in index:
        #先处理app目录
        s += 1
        if s % 500000 ==0:
             print("have been handle size is {0}".format(s))
        app_list = handle_app_str(app[inde])
        re_dict.setdefault(user[inde],app_list)
    print("finish get_user_dict")
    return re_dict

def cov_app_to_cate(app_list,acdict): 
    cate_list = []
    for a in app_list:
        cate_list.append(acdict.get(a))
    return cate_list

def get_nullcate_dict():
    cate = pd.read_csv(r'E:\finaldata\final\app_categories.csv')
    cate = cate.sort_values(by = 'appCategory') 
    ct = cate['appCategory']
    index = list(cate.index) 
    cate_dict = {}
    for i in index: 
        #先构造一个{目录：0}的字典
        if ct[i] not in cate_dict:
            cate_dict.setdefault(ct[i],0) 
    return cate_dict

def get_app_cate_dict():
    cate = pd.read_csv(r'E:\finaldata\final\app_categories.csv')
    cate = cate.sort_values(by = 'appID') 
    ct = cate['appCategory']
    app = cate['appID']
    index = list(cate.index) 
    a_c_dict = {}
    for i in index: 
        #先构造一个{目录：0}的字典
        if app[i] not in cate_dict:
            a_c_dict.setdefault(app[i],0) 
    return a_c_dict


def get_user_fea_from_install(df_h,app_size_down = 30,app_size_up = 60):
    now = time.time()
    user = df_h['userID'] 
    app_dict = get_user_dict(df_h) 
    print("finish production dict in time {0:6.0f}".format(time.time()-now))
    now = time.time()
    index = list(df_h.index)
    _user_cnt = pd.Series(np.zeros(df_h.shape[0]),index = df_h.index)
    _user_isornot_big = pd.Series(np.zeros(df_h.shape[0]),index = df_h.index)
    s = 0
    for i in index:
        s += 1
        if s % 500000 ==0:
             print("have been handle size is {0}".format(s))
        l = len(app_dict.get(user[i]))
        _user_cnt[i] = l
        if l >= app_size_down and l < app_size_up:
            _user_isornot_big[i] = 1
        elif l >= app_size_up:
            _user_isornot_big[i] = 2 
    df_h['cnt_user_app_install'] = _user_cnt
    df_h['isOrNot_big_user'] = _user_isornot_big
    df_h.to_csv(r'E:\finaldata\final\usefea\user_num_or_big.csv',index = False)
    print("finish production dict in time {0:6.0f}".format(time.time()-now))
    now = time.time()
    keys = app_dict.keys()
    #先构造一个{目录：0}的字典
    cate_dict = get_nullcate_dict()
    #构造一个{appID：cate}的字典
    a_c_dict = get_app_cate_dict()
    user_app_cate_dict = {}
    s = 0
    for k in keys: 
        s += 1
        if s % 500000 ==0:
             print("have been handle size is {0}".format(s))
        k_dict = {}
        app_list = app_dict.get(k)
        #形成目录列表
        app_cate_list = cov_app_to_cate(app_list,a_c_dict)
        app_cate_list.sort()
        for acl in app_cate_list:
            if acl not in k_dict:
                k_dict.setdefault(acl,1)
            else:
                k_dict[acl] += 1
        user_app_cate_dict.setdefault(k,k_dict)
    #完成{用户：{cate:cnt}}，转成DataFrame
    u_a_a_pd = pd.DataFrame(user_app_cate_dict).fillna(value = 0).T
    u_a_a_pd.to_csv(r'E:\finaldata\final\usefea\user_catecnt.csv')
    print("finish production dict in time {0:6.0f}".format(time.time()-now))

def df2ffm(df, fp ,prompt_size = 1000000):
    '''
    Convert pandas.DataFrame to data format that libffm can directly use

    @Args:
        df: pandas.DataFrame to be converted
        fp: save libffm format data to fp<filepath>
    '''
    print('Format Converting ...')
    columns = df.columns.values
    d = len(columns)
    feature_index = [i for i in range(d)]
    field_index = [0]*d
    field = []
    for col in columns:
        field.append(col.split('_')[0]) 
    index = -1
    for i in range(d):
        if i==0 or field[i]!=field[i-1]:
            index+=1
        field_index[i] = index 
    with open(fp, 'w') as f:
        num = 0
        for row in df.values:
            num += 1
            if num % prompt_size == 0:
                print("have been write data in size {0}".format(num))
            line = str(row[0])
            for i in range(1, len(row)):
                if row[i]!=0:
                    line += " %d:%d:%d" % (field_index[i], feature_index[i], row[i])
            line+='\n'
            f.write(line)
    print('[Done]')
    print()

def get_df_to_FFMData(df,fp,isOneHotFeaName = [],isOneHot = False):
    '''
    Convert pandas.DataFrame to data format that libffm can directly use
    Field_index:Feature_index:Value
    @Args:
        df: pandas.DataFrame to be converted
        fp: save libffm format data to fp<filepath>
        isOneHot: the feature needs or not needs Onehot
        isOneHotFeaName:the list
    '''
    print("begin handle DataFrame .......")
    _tolTime = time.time()
    if isOneHot == True:
        for fea in isOneHotFeaName:
            now = time.time()
            print("begin one hot feature {0}".format(fea))
            ohe = sklearn.preprocessing.OneHotEncoder()
            ohe_ft  = ohe.fit_transform(df[fea].values.reshape(-1,1))
            ohe_pd = pd.DataFrame(ohe_ft.toarray())
            new_col_list = []
            for col in ohe_pd.columns:
                new_col = str(fea)+ '_' + str(col)
                new_col_list.append(new_col)
            ohe_pd.columns = new_col_list
            df = pd.concat([df,ohe_pd],axis=1)
            del ohe_pd
            print("feature one hot feature {0} in time {1:6.0f}".format(fea,time.time()-now))
    features =[x for x in df.columns if x not in isOneHotFeaName]
    df = df[features]
    print(df.columns)
    df2ffm(df,fp)
    print("finish handle DataFrame to FFMData or LibSVM data in time {0:6.0f}".format(time.time()-_tolTime))

#test: 
'''
train = pd.DataFrame()
for t in range(28,32):
    path = "E:\\finaldata\\final\\splitbyday\\train_"+str(t)
    trian1 = pd.read_pickle(path)
    train = pd.concat([train,trian1],ignore_index=True)
    print(t)
    del trian1 
gc.collect()
'''
train = pd.read_csv(r'E:\finaldata\final\sample\feadata\data.csv')
#'positionID', 'connectionType', 'telecomsOperator', 'gender', 'education', 'marriageStatus', 'haveBaby',
print(train.columns)

features = ['positionID', 'connectionType', 'telecomsOperator','hometown','sitesetID', 'positionType']

for fea in features:
    print("begin feature in {0}".format(fea))
    now = time.time()
    cnt = "cnt_" + fea 
    cov = "cnt_cov_" +fea
    train[cnt],train[cov] = get_cnt_single(train,fea)
    print("finish feature {0} in time {1}".format(fea,time.time()-now))

train.to_csv(r'E:\finaldata\final\sample\feadata\totalData.csv',index = False)

'''
trian = pd.read_pickle(r'E:\finaldata\final\splitdata\train_17')
trian1 = pd.read_pickle(r'E:\finaldata\final\splitdata\train_18')
train = pd.concat([trian,trian1])
del trian
del trian1
gc.collect()
'''
'''
train = pd.DataFrame()
for t in range(17,21):
    path = "E:\\finaldata\\final\\splitbyday\\train_"+str(t)
    trian1 = pd.read_pickle(path)
    train = pd.concat([train,trian1],ignore_index=True)
    print(t)
    del trian1 
gc.collect()
train['_bindApp_fea'] =  get_bindApp_fea(train)
print("---------------------------train.groupby('_bindApp_fea')['label'].mean()----------------")
print(train.groupby('_bindApp_fea')['label'].mean())
print("---------------------------train['_bindApp_fea'].value_counts()----------------")
print(train['_bindApp_fea'].value_counts())
print("---------------------------train['_bindApp_fea'].mean()----------------")
print(train['_bindApp_fea'].mean())
'''
'''
train['_cnt_eveyday_on_user'],train['_cov_everyday_on_user']= get_cntAndCovCnt_every_day(train,day_begin=17,day_end=19)
print("--------------------finish get_cntAndCovCnt_every_day-------------------------------------------------------------------")
train['_totalCovRank'] = get_covRank_sameUserIDClickCnt_in_day(train,pre_cov_cnt='_cov_everyday_on_user',day_begin=17,day_end=19)
train['_covRatio'],train['_covRatioRank'] = get_covRatioAndRank_sameUserIDCreativeID_in_day(train,day_begin=17,day_end=19)
#train['_bindApp_fea'] =  get_bindApp_fea(train)
print(train['_covRatio'].mean())
print(train['_totalCovRank'].mean())
print(train['_covRatioRank'].mean())
#print(train['_covRatio'].mean())
'''
'''
train['_preOrNextClickRank'] = get_preOrNextClickRank_in_day(train,day_begin=17,day_end=19)
#print(train['_preOrNextClickRank'])

train['_eur'] = get_user_ever_num(train)
train['_rank'] = get_total_num_rank_fea(train,pre_num_rank='_eur',every=True)
print(train['_eur'])
print(train['_rank'])
print(train['_preOrNextClickRank'])
print(train['_eur'].mean())
print(train['_rank'].mean())
print(train['_preOrNextClickRank'].mean())
test = train[['_eur','_rank','_preOrNextClickRank']]
test.to_csv(r'E:\finaldata\test1.csv',index = False)
'''