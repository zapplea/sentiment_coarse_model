import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')


import csv
namelist_filePath='/datastore/liu121/senti_data/name_list/name_list_nostop'
files_name = os.listdir(namelist_filePath)
dic={}
for file_name in files_name:
    file_path=namelist_filePath+'/'+file_name
    dic[file_name] =[]
    for row in csv.reader(file_path):
        print(type(row))
        break