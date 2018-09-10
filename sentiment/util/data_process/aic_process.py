import pandas as pd
import jieba

class AiC:
    def __init__(self,config):
        self.config = config

    def reader(self):
        data = pd.read_csv(self.config['train_filePath'])
        print(data[0])


if __name__ == "__main__":
    config = {'train_filePath':'/datastore/liu121/sentidata2/expdata/aic2018/train/sentiment_analysis_trainingset.csv'}
    aic = AiC(config)
    aic.reader()