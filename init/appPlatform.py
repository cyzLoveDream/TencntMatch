# -*- coding: utf-8 -*-
"""
Created on Wed May  3 08:49:38 2017

@author: Administrator
"""

import pymysql
import time
 
print("开始")
now = time.time()
#定义上下文管理器，连接后自动关闭连接
# 创建连接
conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='', db='tencentdata', charset='utf8')
# 创建游标
cursor = conn.cursor()
  
# 执行SQL，并返回收影响行数
effect_row =cursor.execute("SELECT DISTINCT appID FROM trainresultuse") 
appID_lei= cursor.fetchall()
appIDList={}
for i in appID_lei:
    appIDinfo = []
    sql = "SELECT COUNT(*) FROM trainresultuse WHERE appID = '%s'" % i
    cursor.execute(sql)
    appID_count = cursor.fetchone()
    appIDinfo.append(appID_count[0])
    
    sql1="SELECT COUNT(*) FROM trainresultuse WHERE appID = '%s' and label=1 and appPlatform=1" % i
    cursor.execute(sql1)
    a1_count = cursor.fetchone()
    appIDinfo.append(a1_count[0])
    sql2="SELECT COUNT(*) FROM trainresultuse WHERE appID = '%s' and label=1 and appPlatform=2" % i
    cursor.execute(sql2)
    a2_count = cursor.fetchone()
    appIDinfo.append(a2_count[0]) 
    appIDList.setdefault(i[0],appIDinfo)
    aappIDinfo = []
appIDListRate = {}
print("查询时间")
print(time.time()-now)
for j in appID_lei:
    rate=[]
    r0 = appIDList.get(j[0])[1]/appIDList.get(j[0])[0]
    rate.append(r0)
    r1 = appIDList.get(j[0])[2]/appIDList.get(j[0])[0]
    rate.append(r1) 
    appIDListRate.setdefault(j[0],rate)
    rate=[]
print(time.time()-now)  
#写入训练集
for m in appID_lei:
    tu0=(appIDListRate.get(m[0])[0],0,m[0])
    sqlr0 = "UPDATE trainresultuse SET prex23 = '%s',prex24 = '%s' WHERE  appID = '%s' and appPlatform= 1 " % tu0
    effert_rowr0 = cursor.execute(sqlr0)
    #print(effert_rowr0)
    tu1=(0,appIDListRate.get(m[0])[1],m[0])
    sqlr1 = "UPDATE trainresultuse SET prex23 = '%s',prex24 = '%s' WHERE  appID = '%s' and appPlatform= 2 " % tu1
    effert_rowr1 = cursor.execute(sqlr1)
   # print(effert_rowr0) 
    print(effert_rowr0+effert_rowr1)
print("更新训练集时间")
print(time.time()-now) 
#写入测试集
for m in appID_lei:
    tu0=(appIDListRate.get(m[0])[0],0,m[0])
    sqlr0 = "UPDATE testresult SET prex23 = '%s',prex24 = '%s' WHERE  appID = '%s' and appPlatform= 1 " % tu0
    effert_rowr0 = cursor.execute(sqlr0)
    #print(effert_rowr0)
    tu1=(0,appIDListRate.get(m[0])[1],m[0])
    sqlr1 = "UPDATE testresult SET prex23 = '%s',prex24 = '%s' WHERE  appID = '%s' and appPlatform= 2 " % tu1
    effert_rowr1 = cursor.execute(sqlr1)
   # print(effert_rowr0) 
    print(effert_rowr0+effert_rowr1)
print("更新测试集时间")
print(time.time()-now) 

print(effect_row)
   
conn.commit()
# 关闭游标
cursor.close()
# 关闭连接
conn.close()
 
