import sys
sys.path.append('/datastore/liu121/py-package/jieba')

import pandas as pd
import jieba
import re
import pickle

class AiC:
    def __init__(self,config):
        self.config = config

    def reader(self):
        data = pd.read_csv(self.config['train_filePath'])
        self.columns_name = data.columns.values
        print('columns name: ',self.columns_name)
        self.train_data = data.values
        print('train shape: ',self.train_data.shape)
        data = pd.read_csv(self.config['testa_filePath'])
        self.testa_data = data.values
        print('test shape: ',self.testa_data.shape)
        data = pd.read_csv(self.config['valid_filePath'])
        self.valid_data = data.values
        print('valid shape: ',self.valid_data.shape)


    def split_reivew(self,paragraph):
        for sent in re.findall(u'[^!?。\.\!\?]+[!?。\.\!\?]?', paragraph, flags=re.U):
            yield sent

    def cut(self,sentence):
        return jieba.cut(sentence,HMM=True)

def write(file_path,data):
    with open(file_path,'wb') as f:
        pickle.dump(data,f)

def main():
    config = {'train_filePath': '/datastore/liu121/sentidata2/expdata/aic2018/train/sentiment_analysis_trainingset.csv',
              'testa_filePath': '/datastore/liu121/sentidata2/expdata/aic2018/testa/sentiment_analysis_testa.csv',
              'valid_filePath': '/datastore/liu121/sentidata2/expdata/aic2018/valid/sentiment_analysis_validationset.csv',
              'pkl_filePath':'/datastore/liu121/sentidata2/expdata/aic2018/data.pkl'}
    aic = AiC(config)
    aic.reader()
    dic = {'train_data':aic.train_data,'testa_data':aic.testa_data,'valid_data':aic.valid_data}

    pkl_dict = {}
    for data_name in dic:
        dataset = dic[data_name]
        pkl_dict[data_name]=[]
        for row in dataset:
            reivew = row[1][1:-1]
            labels = row[2:]
            sentences = list(aic.split_reivew(reivew))
            for sentence in sentences:
                pkl_dict[data_name].append((aic.cut(sentence),labels))
    write(config['pkl_filePath'],pkl_dict)


if __name__ == "__main__":
    main()