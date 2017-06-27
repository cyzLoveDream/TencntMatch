# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:56:35 2017

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

for i in range(25,42,1):
    str1 = "prex" + str(i)
    sql = "ALTER TABLE trainresultuse ADD %s VARCHAR(255)" % str1
    cursor.execute(sql)
    effct_row = cursor.fetchone()
    print(i)
conn.commit()
# 关闭游标
cursor.close()
# 关闭连接
conn.close()
 