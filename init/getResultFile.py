# -*- coding: utf-8 -*-
"""
Created on Tue May  2 20:41:19 2017

@author: Administrator
"""

import csv
import time
now = time.time()
with open(r'E:\tencent\output\result1.csv') as reFlie:
     l=csv.reader(reFlie)
     for line in l:
         if line[0] == 'instance':
             if line[1] == 'prod':
                 f = open(r'E:\tencent\output\submission.csv','a+')
                 f.write(line[0]+","+line[1]+"\n") 
                 continue
         l=int(line[0])
         l +=1
         line[0] = str(l)
       #  line[1] = int(round(line[1],8))
        # line[1] = abs(line[1])
        # line[1] = str(line[1])
         f.write(line[0]+","+line[1]+"\n")
f.close()
print(time.time()-now)
                    
