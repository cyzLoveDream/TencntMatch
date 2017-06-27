from collections import defaultdict
import marshal
import utils
from utils import *

from csv import DictWriter

train_path = "E:\\finaldata\\final\\splitdata\\train_17"
test_path = "E:\\finaldata\\final\\splitdata\\train_31"
train_output = "E:\\finaldata\\final\\splitdata\\train_17_rank"
test_output = "E:\\finaldata\\final\\splitdata\\train_31_rank"

keys_raw = ['label','creativeID','userID','positionID','connectionType','telecomsOperator']
keys_new = ['label','day','hour','mm','creativeID','userID','positionID','connectionType','telecomsOperator',
'user_rank','user_d_rank','user_h_rank','user_ad_rank','user_ad_d_rank','user_ad_h_rank']

def run():
	count = 0
	# all rank feature
	user_cnt = defaultdict(int)
	user_d_cnt = defaultdict(int)
	user_h_cnt = defaultdict(int)
	
	# related to creativeID
	user_ad_cnt = defaultdict(list)
	user_ad_d_cnt = defaultdict(list)
	user_ad_h_cnt = defaultdict(list)
	
	
	with open(train_output,'w') as f:
		writer = DictWriter(f,fieldnames = keys_new)
		writer.writeheader()
		for t, row in enumerate(DictReader(open(train_path), delimiter=',')):
			count += 1
			if count % 100000 == 0:
				print(count)
			new_row = {}
			for key in keys_raw:
				new_row[key] = row[key]
			
			day = int(int(row['clickTime'])/10000)
			hour = 24*day + int(int(row['clickTime'])%10000)
			new_row['day'] = day 
			new_row['hour'] = hour
			userID = row['userID']
			creativeID = row['creativeID']
			
			# camID_cnt_by_aderID[row['adver']] += 1
			# adID_cnt_by_camID += 1
			# creID_cnt_by_adID += 1
			# camID_cnt_by_appID += 1

			user_cnt[userID] += 1
			user_d_cnt[str(day)+str(userID)] += 1
			user_h_cnt[str(hour)+str(userID)] += 1

			user_ad_cnt[userID].append(creativeID)
			user_ad_d_cnt[str(day)+str(userID)].append(creativeID)
			user_ad_h_cnt[str(hour)+str(userID)].append(creativeID)
			

			new_row['user_rank'] = user_cnt[userID]
			new_row['user_d_rank'] = user_d_cnt[str(day)+str(userID)]
			new_row['user_h_rank'] = user_h_cnt[str(hour)+str(userID)]

			new_row['user_ad_rank']  = len(user_ad_cnt[userID])
			new_row['user_ad_d_rank'] = len(user_ad_d_cnt[str(day)+str(userID)])
			new_row['user_ad_h_rank'] = len(user_ad_h_cnt[str(hour)+str(userID)])

			writer.writerow(new_row)
	f.close()
	with open(test_output,'w') as f:
		writer = DictWriter(f,fieldnames = ['instanceID'] + keys_new)
		writer.writeheader()
		for t, row in enumerate(DictReader(open(test_path), delimiter=',')):
			count += 1
			if count % 100000 == 0:
				print(count)
			new_row = {}
			for key in ['instanceID'] + keys_raw:
				new_row[key] = row[key]
			day = int(int(row['clickTime'])/10000)
			hour = 24*day + int(int(row['clickTime'])%10000)
			new_row['day'] = day 
			new_row['hour'] = hour
			userID = row['userID']
			creativeID = row['creativeID']
			if userID in user_cnt.keys():
				user_cnt[userID] += 1
			else:
				user_cnt[userID] = 1
			user_d_cnt[str(day)+str(userID)] += 1
			user_h_cnt[str(hour)+str(userID)] += 1

			user_ad_cnt[userID].append(creativeID)
			user_ad_d_cnt[str(day)+str(userID)].append(creativeID)
			user_ad_h_cnt[str(hour)+str(userID)].append(creativeID)

			new_row['user_rank'] = user_cnt[userID]
			new_row['user_d_rank'] = user_d_cnt[str(day)+str(userID)]
			new_row['user_h_rank'] = user_h_cnt[str(hour)+str(userID)]

			new_row['user_ad_rank']  = len(user_ad_cnt[userID])
			new_row['user_ad_d_rank'] = len(user_ad_d_cnt[str(day)+str(userID)])
			new_row['user_ad_h_rank'] = len(user_ad_h_cnt[str(hour)+str(userID)])

			writer.writerow(new_row)
	f.close()
	write_dump('../data/user_cnt.dump',user_cnt)
	write_dump('../data/user_h_cnt.dump',user_h_cnt)
	write_dump('../data/user_d_cnt.dump',user_d_cnt)
	write_dump('../data/user_ad_cnt.dump',user_ad_cnt)
	write_dump('../data/user_ad_d_cnt.dump',user_ad_d_cnt)
	write_dump('../data/user_ad_h_cnt.dump',user_ad_h_cnt)

run() 
