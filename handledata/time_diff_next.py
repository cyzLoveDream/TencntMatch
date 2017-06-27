import csv, sys, collections, time, os

FIELDS = ['label', 'clickTime', 'conversionTime', 'creativeID', 'userID','positionID', 'connectionType', 'telecomsOperator']
NEW_FIELDS = ['instanceID'] + FIELDS+['user_time_diff','user_creative_time_diff']

time_diff = collections.defaultdict(lambda: {'clickTime_prev': ''})
creative_time_diff = collections.defaultdict(lambda: {'clickTime_prev': ''})
start = time.time()

def reversed_lines(file):
    "Generate the lines of file in reverse order."
    part = ''
    for block in reversed_blocks(file):
        for c in reversed(block):
            if c == '\n' and part:
                yield part[::-1]
                part = ''
            part += c
    if part: yield part[::-1]

def reversed_blocks(file, blocksize=4096):
    "Generate blocks of file's contents in reverse order."
    file.seek(0, os.SEEK_END)
    here = file.tell()
    while 0 < here:
        delta = min(blocksize, here)
        here -= delta
        file.seek(here, os.SEEK_SET)
        yield file.read(delta)


def gen_data(src_path, dst_app_path):
    reader1 = csv.DictReader(open(src_path))
    reader2 = csv.DictReader(open(src_path))
    writer = csv.DictWriter(open(dst_app_path, 'w'), NEW_FIELDS)
    writer.writeheader()

    for i, row in enumerate(csv.reader(reversed_lines(open(src_path))), start=1):
        if i % 1000000 == 0:
            sys.stderr.write('{0:6.0f}    {1}m\n'.format(time.time()-start,int(i/1000000)))
        
        new_row = {}
        new_row['instanceID'] = i
        for field,i in zip(FIELDS,range(len(FIELDS))):
            new_row[field] = row[i]

        # 源文件不一样，可能index不同，我这里是按照train文件的格式来的
        userID, creativeID, clickTime =  row[4], row[3], row[1]
        if time_diff[userID]['clickTime_prev'] == '':
            new_row['user_time_diff'] = -1 
        else:
            new_row['user_time_diff'] = int(time_diff[userID]['clickTime_prev']) - int(clickTime)
        time_diff[userID]['clickTime_prev'] = clickTime

        if creative_time_diff[str(userID)+str(creativeID)]['clickTime_prev'] == '':
            new_row['user_creative_time_diff'] = -1
        else:
            new_row['user_creative_time_diff'] = int(creative_time_diff[str(userID)+str(creativeID)]['clickTime_prev']) - int(clickTime)
        creative_time_diff[str(userID)+str(creativeID)]['clickTime_prev'] = clickTime

        writer.writerow(new_row)

gen_data(utils.data_path + '/train.csv', utils.data_path + '/train_time_diff.csv')
