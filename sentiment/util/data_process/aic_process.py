import csv

class AiC:
    def __init__(self,config):
        self.config = config

    def reader(self):
        with open(self.config['train_filePath'], newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in data:
                print(row)
                exit()


if __name__ == "__main__":
    config = {'train_filePath':'/datastore/liu121/sentidata2/expdata/aic2018/train'}
    aic = AiC(config)
    aic.reader()