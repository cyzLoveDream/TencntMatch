import argparse, csv, sys, pickle, collections, time
import utils

FIELDS = ['label', 'clickTime', 'conversionTime', 'creativeID', 'userID','positionID', 'connectionType', 'telecomsOperator','day_hour','day','hour']
NEW_FIELDS = FIELDS+['user_click_histroy']

history = collections.defaultdict(lambda: {'history': '', 'buffer': '', 'prev_hour': ''})

start = time.time()


def gen_data(src_path, dst_app_path):
    reader = csv.DictReader(open(src_path))
    writer = csv.DictWriter(open(dst_app_path, 'w'), NEW_FIELDS)
    writer.writeheader()

    for i, row in enumerate(reader, start=1):
        if i % 1000000 == 0:
            sys.stderr.write('{0:6.0f}    {1}m\n'.format(time.time()-start,int(i/1000000)))
        
        new_row = {}
        for field in FIELDS:
            new_row[field] = row[field]

        user, hour =  row['userID'], row['day_hour']
        new_row['user_count'] = user_cnt[user]
        if history[user]['prev_hour'] != row['day_hour']:
            history[user]['history'] = (history[user]['history'] + history[user]['buffer'])[-4:]
            history[user]['buffer'] = ''
            history[user]['prev_hour'] = row['day_hour']

        new_row['user_click_histroy'] = history[user]['history']

        history[user]['buffer'] += row['label']

        writer.writerow(new_row)
        


print('======================scan complete======================')

gen_data(utils.data_path + '/train_0604.csv', utils.data_path + '/train_history.csv')
