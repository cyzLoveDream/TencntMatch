# -*- coding: utf-8 -*-
"""
Created on Sun May 14 12:06:03 2017

@author: Administrator
"""

import pandas as pd 
import numpy as np

df = pd.read_csv(r'E:\tencent\pre\appidandcategories.csv')
ac = df['appCategory'].value_counts()
ac_log = ac.apply(lambda x: np.log(x+1))
ac_list = df['appCategory'].tolist()
append_list = []
print('开始处理')
a=0
for i in ac_list:
    a +=1
    append_list.append(ac_log[i])
    if a%1000000==0:
        print(a)
print('处理完成')
append_list = np.array(append_list)
ap = pd.DataFrame(append_list)
dfres = pd.contact([df,ap],axis = 1)
dfres.to_csv(r'E:\tencent\pre\appidandcategorieslog.csv',index = False)