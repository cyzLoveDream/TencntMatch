 # -*- coding: utf-8 -*-
"""
Created on Mon May  1 16:37:24 2017

@author: Administrator
"""

import pymysql
import time
 
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
    sql0="SELECT COUNT(*) FROM trainresultuse WHERE appID = '%s' and label=1 and haveBaby=0" % i
    cursor.execute(sql0)
    a0_count = cursor.fetchone()
    appIDinfo.append(a0_count[0])
    sql1="SELECT COUNT(*) FROM trainresultuse WHERE appID = '%s' and label=1 and haveBaby=1" % i
    cursor.execute(sql1)
    a1_count = cursor.fetchone()
    appIDinfo.append(a1_count[0])
    sql2="SELECT COUNT(*) FROM trainresultuse WHERE appID = '%s' and label=1 and haveBaby=2" % i
    cursor.execute(sql2)
    a2_count = cursor.fetchone()
    appIDinfo.append(a2_count[0])
    sql3="SELECT COUNT(*) FROM trainresultuse WHERE appID = '%s' and label=1 and haveBaby=3" % i
    cursor.execute(sql3)
    a3_count = cursor.fetchone()
    appIDinfo.append(a3_count[0])
    sql4="SELECT COUNT(*) FROM trainresultuse WHERE appID = '%s' and label=1 and haveBaby=4" % i
    cursor.execute(sql4)
    a4_count = cursor.fetchone()
    appIDinfo.append(a4_count[0])
    sql5="SELECT COUNT(*) FROM trainresultuse WHERE appID = '%s' and label=1 and haveBaby=5" % i
    cursor.execute(sql5)
    a5_count = cursor.fetchone()
    appIDinfo.append(a5_count[0])
    sql6="SELECT COUNT(*) FROM trainresultuse WHERE appID = '%s' and label=1 and haveBaby=6" % i
    cursor.execute(sql6)
    a6_count = cursor.fetchone()
    appIDinfo.append(a6_count[0]) 
    appIDList.setdefault(i[0],appIDinfo)
    aappIDinfo = []
appIDListRate = {}
print("查询时间")
print(time.time()-now)
now = time.time()
for j in appID_lei:
    rate=[]
    r0 = appIDList.get(j[0])[1]/appIDList.get(j[0])[0]
    rate.append(r0)
    r1 = appIDList.get(j[0])[2]/appIDList.get(j[0])[0]
    rate.append(r1)
    r2 = appIDList.get(j[0])[3]/appIDList.get(j[0])[0]
    rate.append(r2)
    r3 = appIDList.get(j[0])[4]/appIDList.get(j[0])[0]
    rate.append(r3)
    r4 = appIDList.get(j[0])[5]/appIDList.get(j[0])[0]
    rate.append(r4)
    r5 = appIDList.get(j[0])[6]/appIDList.get(j[0])[0]
    rate.append(r5)
    r6 = appIDList.get(j[0])[7]/appIDList.get(j[0])[0]
    rate.append(r6) 
    appIDListRate.setdefault(j[0],rate)
    rate=[]
print(time.time()-now) 
now = time.time() 
#写入训练集
for m in appID_lei:
    tu0=(appIDListRate.get(m[0])[0],0,0,0,0,0,0,m[0])
    sqlr0 = "UPDATE trainresultuse SET prex16 = '%s',prex17 = '%s',prex18 = '%s',prex19 = '%s',prex20 = '%s',prex21 = '%s' , prex22 = '%s' WHERE  appID = '%s' and haveBaby= 0 " % tu0
    effert_rowr0 = cursor.execute(sqlr0)
    #print(effert_rowr0)
    tu1=(0,appIDListRate.get(m[0])[1],0,0,0,0,0,m[0])
    sqlr1 = "UPDATE trainresultuse SET prex16 = '%s',prex17 = '%s',prex18 = '%s',prex19 = '%s',prex20 = '%s',prex21 = '%s' , prex22 = '%s' WHERE  appID = '%s' and haveBaby= 1 " % tu1
    effert_rowr1 = cursor.execute(sqlr1)
   # print(effert_rowr0)
    tu2= (0,0,appIDListRate.get(m[0])[2],0,0,0,0,m[0])
    sqlr2 = "UPDATE trainresultuse SET prex16 = '%s',prex17 = '%s',prex18 = '%s',prex19 = '%s',prex20 = '%s',prex21 = '%s' , prex22 = '%s' WHERE  appID = '%s' and haveBaby= 2 " % tu2
    effert_rowr2 = cursor.execute(sqlr2)
    tu3=(0,0,0,appIDListRate.get(m[0])[3],0,0,0,m[0])
    sqlr3 = "UPDATE trainresultuse SET prex16 = '%s',prex17 = '%s',prex18 = '%s',prex19 = '%s',prex20 = '%s',prex21 = '%s' , prex22 = '%s' WHERE  appID = '%s' and haveBaby= 3 " % tu3
    effert_rowr3 = cursor.execute(sqlr3)
    tu4=(0,0,0,0,appIDListRate.get(m[0])[4],0,0,m[0])
    sqlr4 = "UPDATE trainresultuse SET prex16 = '%s',prex17 = '%s',prex18 = '%s',prex19 = '%s',prex20 = '%s',prex21 = '%s' , prex22 = '%s' WHERE  appID = '%s' and haveBaby= 4 " % tu4
    effert_rowr4 = cursor.execute(sqlr4)
    tu5=(0,0,0,0,0,appIDListRate.get(m[0])[5],0,m[0])
    sqlr5 = "UPDATE trainresultuse SET prex16 = '%s',prex17 = '%s',prex18 = '%s',prex19 = '%s',prex20 = '%s',prex21 = '%s' , prex22 = '%s' WHERE  appID = '%s' and haveBaby= 5 " % tu5
    effert_rowr5 = cursor.execute(sqlr5)
    tu6=(0,0,0,0,0,0,appIDListRate.get(m[0])[6],m[0])
    sqlr6 = "UPDATE trainresultuse SET prex16 = '%s',prex17 = '%s',prex18 = '%s',prex19 = '%s',prex20 = '%s',prex21 = '%s' , prex22 = '%s' WHERE  appID = '%s' and haveBaby= 6 " % tu6
    effert_rowr6 = cursor.execute(sqlr6) 
    print(effert_rowr0+effert_rowr1+effert_rowr2+effert_rowr3+effert_rowr4+effert_rowr5+effert_rowr6)
print("更新训练集时间")
print(time.time()-now) 
now = time.time()
#写入testresult
for m in appID_lei:
    tu0=(appIDListRate.get(m[0])[0],0,0,0,0,0,0,m[0])
    sqlr0 = "UPDATE testresult SET prex16 = '%s',prex17 = '%s',prex18 = '%s',prex19 = '%s',prex20 = '%s',prex21 = '%s' , prex22 = '%s' WHERE  appID = '%s' and haveBaby= 0 " % tu0
    effert_rowr0 = cursor.execute(sqlr0)
    #print(effert_rowr0)
    tu1=(0,appIDListRate.get(m[0])[1],0,0,0,0,0,m[0])
    sqlr1 = "UPDATE testresult SET prex16 = '%s',prex17 = '%s',prex18 = '%s',prex19 = '%s',prex20 = '%s',prex21 = '%s' , prex22 = '%s' WHERE  appID = '%s' and haveBaby= 1 " % tu1
    effert_rowr1 = cursor.execute(sqlr1)
   # print(effert_rowr0)
    tu2= (0,0,appIDListRate.get(m[0])[2],0,0,0,0,m[0])
    sqlr2 = "UPDATE testresult SET prex16 = '%s',prex17 = '%s',prex18 = '%s',prex19 = '%s',prex20 = '%s',prex21 = '%s' , prex22 = '%s' WHERE  appID = '%s' and haveBaby= 2 " % tu2
    effert_rowr2 = cursor.execute(sqlr2)
    tu3=(0,0,0,appIDListRate.get(m[0])[3],0,0,0,m[0])
    sqlr3 = "UPDATE testresult SET prex16 = '%s',prex17 = '%s',prex18 = '%s',prex19 = '%s',prex20 = '%s',prex21 = '%s' , prex22 = '%s' WHERE  appID = '%s' and haveBaby= 3 " % tu3
    effert_rowr3 = cursor.execute(sqlr3)
    tu4=(0,0,0,0,appIDListRate.get(m[0])[4],0,0,m[0])
    sqlr4 = "UPDATE testresult SET prex16 = '%s',prex17 = '%s',prex18 = '%s',prex19 = '%s',prex20 = '%s',prex21 = '%s' , prex22 = '%s' WHERE  appID = '%s' and haveBaby= 4 " % tu4
    effert_rowr4 = cursor.execute(sqlr4)
    tu5=(0,0,0,0,0,appIDListRate.get(m[0])[5],0,m[0])
    sqlr5 = "UPDATE testresult SET prex16 = '%s',prex17 = '%s',prex18 = '%s',prex19 = '%s',prex20 = '%s',prex21 = '%s' , prex22 = '%s' WHERE  appID = '%s' and haveBaby= 5 " % tu5
    effert_rowr5 = cursor.execute(sqlr5)
    tu6=(0,0,0,0,0,0,appIDListRate.get(m[0])[6],m[0])
    sqlr6 = "UPDATE testresult SET prex16 = '%s',prex17 = '%s',prex18 = '%s',prex19 = '%s',prex20 = '%s',prex21 = '%s' , prex22 = '%s' WHERE  appID = '%s' and haveBaby= 6 " % tu6
    effert_rowr6 = cursor.execute(sqlr6) 
    print(effert_rowr0+effert_rowr1+effert_rowr2+effert_rowr3+effert_rowr4+effert_rowr5+effert_rowr6)
print("更新测试集时间")
print(time.time()-now)
print(effect_row)
   
conn.commit()
# 关闭游标
cursor.close()
# 关闭连接
conn.close()
 

