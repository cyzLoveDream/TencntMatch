# -*- coding: utf-8 -*-
"""
Created on Fri May 26 20:57:32 2017

@author: Administrator
"""
import pandas as pd
import os

i=0
for file in os.listdir(r'E:\tencent\input\spilitnoonehotdata'):
    i+=1
    filepath = "E:\\tencent\\input\\spilitnoonehotdata\\"+ str(i) 
    initdata = pd.read_pickle(filepath)
    mfilepath = "E:\\tencent\\input\\spilitnoonehotdata\\" + str(i+1)
    mdata = pd.read_pickle(mfilepath)
    attrs = ['positionID','connectionType','telecomsOperator','age','hometown','residence','advertiserID','appID','appPlatform','sitesetID','positionType'
             ,'adID','camgaignID','appCategory']
    for att in attrs:
        file_dir = "E:\\tencent\\input\\splitfeature\\"
        inner_data = initdata.groupby(att).size() 
        inner_data = pd.DataFrame(inner_data)
        columns = inner_data.columns
        column = []
        for col in columns:
            col = att +"_"+ str(col)
            column.append(col)
        inner_data.columns = column
        file_path = file_dir + att + ".txt" 
        inner_data.to_csv(file_path)
        in_data = pd.read_csv(file_path) 
        mdata = mdata.merge(in_data,how='left',on= att)  
    '''
    combine_first=['positionID','camgaignID']
    conmbine_second = ['connectionType','age','gender','marriageStatus','advertiserID','hometown',
                       'telecomsOperator','creativeID','education','residence','haveBaby']
    for first in combine_first:
        for second in conmbine_second:
            combine = initdata.groupby([first,second,'label']).size().unstack()
            combine.to_csv(r'E:\tencent\input\spilitnoonehotdata\result\huancun\combine.txt')
            combine = pd.read_csv(r'E:\tencent\input\spilitnoonehotdata\result\huancun\combine.txt')
            combine.fillna(value = 0,inplace = True)
            firstcol = first+ '_' + second + "_" + str(0) 
            secondcol = first+'_' + second + "_" + str(1)
            combine.columns = [first,second,firstcol,secondcol]
            mdata = mdata.merge(combine,how='left',on=[first,second])
            combine['row_sum'] = combine.iloc[:,2:].apply(lambda x : x.sum(),axis=1)
            combine[firstcol] = combine[firstcol] / combine['row_sum']
            combine[secondcol] = combine[secondcol] / combine['row_sum']
            combine.drop('row_sum',axis=1,inplace=True)
            firstcol = first +'-' + second + "-" + str(0) 
            secondcol = first +'-'+ second + "-" + str(1)
            combine.columns = [first,second,firstcol,secondcol]
            mdata = mdata.merge(combine,how='left',on=[first,second])  
    '''
    filedir = "E:\\tencent\\input\\spilitnoonehotdata\\result\\"+str(i+1)
    mdata.to_pickle(filedir) 
  #  print(count)
    print(i)

train_data = pd.read_pickle(r'E:\tencent\input\spilitnoonehotdata\result\2')
i=2
for file in os.listdir(r'E:\tencent\input\spilitnoonehotdata\result'):    
    i+=1    
    filename = "E:\\tencent\\input\\spilitnoonehotdata\\result\\"+str(i)      
    print(i)  
    if(i>10):
        continue
    data = pd.read_pickle(filename)    
    train_data = train_data.append(data)
train_data.to_pickle(r'E:\tencent\input\lr\train')
