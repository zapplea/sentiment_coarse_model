import sys
sys.path.append('/datastore/liu121/py-package/jieba')

import pandas as pd
import jieba

class AiC:
    def __init__(self,config):
        self.config = config

    def reader(self):
        data = pd.read_csv(self.config['train_filePath'])
        train_data = data.values
        print('train shape: ',train_data.shape)
        data = pd.read_csv(self.config['testa_filePath'])
        testa_data = data.values
        print('test shape: ',testa_data.shape)
        data = pd.read_csv(self.config['valid_filePath'])
        valid_data = data.values
        print('valid shape: ',valid_data.shape)


    def split_sentence(self,sentence):



if __name__ == "__main__":
    config = {'train_filePath':'/datastore/liu121/sentidata2/expdata/aic2018/train/sentiment_analysis_trainingset.csv',
              'testa_filePath':'/datastore/liu121/sentidata2/expdata/aic2018/testa/sentiment_analysis_testa.csv',
              'valid_filePath':'/datastore/liu121/sentidata2/expdata/aic2018/valid/sentiment_analysis_validationset.csv',}
    aic = AiC(config)
    aic.reader()