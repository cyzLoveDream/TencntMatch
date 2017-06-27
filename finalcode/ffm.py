
from libffmWrapper import FFM
import pandas as pd
import gc

data = pd.read_csv(r'E:\finaldata\final\sample\feadata\data.csv')
train_data =data[data['day']<=30]
test_data = data[data['day']==31] 
print("finish data input")
del data
gc.collect()
print(1) 
#处理空值
test_data = test_data.fillna(value=0)
train_data = train_data.fillna(value=0) 
features = [x for x in train_data.columns if x not in ['label','instanceID','day','clickTime','userID','conversionTime','_bindApp_fea']] 
val_data = train_data[train_data['day']==30]
train_data = train_data[train_data['day']!=30] 

libffm_home = "D:\\ProgramData\\libffm-1.13"    # libffm home

ffm = FFM(libffm_home) 

print("begin train")

ffm.train(train_data,val_data,iteration=200,auto_stop=True,on_disk=True,nr_threads=7,no_rand=True)    

print("begin product")

ffm.predict(test_data,r'E:\finaldata\result\submission.csv')
