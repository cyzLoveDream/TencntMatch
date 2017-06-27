import pandas as pd 
import numpy as np 
data = pd.read_csv('../test/data_test.csv')
data = data.sort_values(['userID','clickTime'])

data['next_clickTime'] = data['clickTime'].shift(-1)
data['_userID'] = data['userID'].shift(-1)
data['duration_next'] = np.where(data['userID'] == data['_userID'], data['next_clickTime'] - data['clickTime'], -1)
print(data.shape)

data = data.sort_values('instanceID')

data.to_csv('../test/test1.csv',index=False)
