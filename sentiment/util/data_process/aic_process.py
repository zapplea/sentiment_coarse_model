import csv

class AiC:
    def __init__(self,config):
        self.config = config

    def reader(self):
        with open(self.config['train_filePath'], newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=' ', quotechar='|')
            count = 0
            for row in data:
                print(row)
                count+=1
                if count==20:
                    exit()


if __name__ == "__main__":
    config = {'train_filePath':'/datastore/liu121/sentidata2/expdata/aic2018/train/sentiment_analysis_trainingset.csv'}
    aic = AiC(config)
    aic.reader()