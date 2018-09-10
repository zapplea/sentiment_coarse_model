import sys
sys.path.append('/datastore/liu121/py-package/jieba')

import pandas as pd
import jieba
import re

class AiC:
    def __init__(self,config):
        self.config = config

    def reader(self):
        data = pd.read_csv(self.config['train_filePath'])
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


if __name__ == "__main__":
    config = {'train_filePath':'/datastore/liu121/sentidata2/expdata/aic2018/train/sentiment_analysis_trainingset.csv',
              'testa_filePath':'/datastore/liu121/sentidata2/expdata/aic2018/testa/sentiment_analysis_testa.csv',
              'valid_filePath':'/datastore/liu121/sentidata2/expdata/aic2018/valid/sentiment_analysis_validationset.csv',}
    aic = AiC(config)
    aic.reader()
    reivew = aic.train_data[0][1]
    sentences = list(aic.split_reivew(reivew))
    print(sentences)