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

for k in [10,15,20]:
    dic={}
    f=open('top'+str(k),'w')
    for file_name in files_name:
        file_path=namelist_filePath+'/'+file_name
        dic[file_name] =[]
        print(file_name)
        with open(file_path,newline='') as csvfile:
            data = csv.reader(csvfile)
            for row in data:
                dic[file_name].append(row[0])
        f.write(file_name+'\n'+str(dic[file_name][:k])+'\n')