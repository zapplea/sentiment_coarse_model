import os
import sys
import pwd
if pwd.getpwuid(os.getuid()).pw_name == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif pwd.getpwuid(os.getuid()).pw_name == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif pwd.getpwuid(os.getuid()).pw_name == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')


import csv
namelist_filePath='/datastore/liu121/senti_data/name_list/name_list_nostop'
files_name = os.listdir(namelist_filePath)
dic={}
for file_name in files_name:
    file_path=namelist_filePath+'/'+file_name
    dic[file_name] =[]
    for row in csv.reader(file_path):
        print(row)
        print(type(row[0]),' ',type(row[1]))
        break
    break