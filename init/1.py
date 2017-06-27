# -*- coding: utf-8 -*-
"""
Created on Wed May 10 22:16:34 2017

@author: Administrator
"""

import pandas as pd 
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn_pandas import DataFrameMapper
import numpy as np
#import numpy as np
#E:\tencent\pre\testresult.csv
#E:\tencent\localTest\localtrain.csv
testData = pd.read_csv(r'E:\tencent\allmerge\testresult.csv')
#testData = np.array(testData)
allData = testData.iloc[:,3:] 
testData = testData.fillna(0)
mapper = DataFrameMapper([
        ('connectionType',LabelBinarizer()),
        ('telecomsOperator',LabelBinarizer()),
        ('education',LabelBinarizer()),
        ('gender',LabelBinarizer()),
        ('marriageStatus',LabelBinarizer()),
        ('haveBaby',LabelBinarizer()),
        ('appPlatform',LabelBinarizer()),
        ('sitesetID',LabelBinarizer()),
        ('positionType',LabelBinarizer()),
        (['age'],[MinMaxScaler(),StandardScaler()]),
        (['appCategory'],OneHotEncoder()),
        ],None)
testData = mapper.fit_transform(allData)
#print(testData)
#print("我是来测试的")
