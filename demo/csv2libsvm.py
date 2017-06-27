# -*- coding: utf-8 -*-
"""
Created on Sat May 13 21:50:03 2017

@author: Administrator
"""
import pandas as pd 
import time
def df2ffm(df, fp):
        '''
        Convert pandas.DataFrame to data format that libffm can directly use

        @Args:
            df: pandas.DataFrame to be converted
            fp: save libffm format data to fp<filepath>
        '''
        now = time.time()
        print('Format Converting begin in time:...',now)
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
            for row in df.values:
                line =str(int(row[0]))
                for i in range(1, len(row)):
                    line += " %s:%s:%d" % (field[i],str(feature_index[i]), row[i])
                line+='\n'
                f.write(line)
        print('finish convert,the cost time is ',time.time()-now)
        print('[Done]')
        print()
        
def main():
    df = pd.read_csv(r'E:\tencent\input\train1.csv') 
    df = df.fillna(0) 
    fp = r'E:\tencent\output\train.csv' 
    df2ffm(df,fp)

if __name__ == '__main__':
    main()
