# -*- coding: utf-8 -*-
"""
Created on Mon May  8 18:11:07 2017

@author: Administrator
"""

import pandas as pd 
import time

now = time.time()
user_appReader = pd.read_csv(r'C:\Users\Administrator\Desktop\pre\user_installedapps.csv')
user_appCon = pd.read_csv(r'C:\Users\Administrator\Desktop\pre\app_categories.csv')

user_appReader = user_appReader.merge(user_appCon,how = 'left',on='appID')

user_appReader.to_csv(r'C:\Users\Administrator\Desktop\pre\appAndCon.csv')
print(now-time.time())