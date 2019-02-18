import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

from aic.data_process.senti_datafeeder import DataFeeder
import numpy as np

def stats(config):
    df = DataFeeder(config)
    # shape = (batch size, attributes num, senti num)
    senti_labels = df.train_senti_labels[:,:20,:]
    shape = np.shape(senti_labels)
    print(shape)
    exit()
    freq={'pos':0,'neg':0,'neu':0,'nmt':0,'pos_neu':0,'pos_neg':0,'neg_neu':0,'neg_pos_neu':0}
    for i in range(shape[0]):
        result = np.sum(senti_labels[i],axis=0)
        condition = np.greater(result,0)
        if np.any(condition):
            if condition[0] and not condition[1] and not condition[2]:
                freq['pos'] += 1
            elif not condition[0] and condition[1] and not condition[2]:
                freq['neu'] +=1
            elif not condition[0] and not condition[1] and condition[2]:
                freq['neg'] +=1
            elif condition[0] and condition[1] and not condition[2]:
                freq['pos_neu'] +=1
            elif condition[0] and not condition[1] and condition[2]:
                freq['pos_neg'] +=1
            elif not condition[0] and condition[1] and condition[2]:
                freq['neg_neu'] +=1
            elif np.all(condition):
                freq['neg_pos_neu'] +=1
        else:
            freq['nmt']+=1
    sum_freq = 0
    for key in freq:
        sum_freq+=freq[key]
    for key in freq:
        print('%s: %.4f'%(key,freq[key]/sum_freq))

if __name__ == "__main__":
    config = {}
    config['train_data_file_path'] = '/datastore/liu121/sentidata2/data/aic2018/coarse_data/train_coarse.pkl'
    config['test_data_file_path'] = '/datastore/liu121/sentidata2/data/aic2018/coarse_data/dev_coarse.pkl'
    stats(config)