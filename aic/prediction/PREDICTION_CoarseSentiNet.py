import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import argparse

from aic.coarse_net.senti_net import SentimentNet
from aic.data_process.attr_datafeeder import DataFeeder
from aic.functions.prediction import SentiPrediction

def main(config):
    datafeeder = DataFeeder(config)
    model_dic = SentimentNet.build(config)
    pred = SentiPrediction(config,datafeeder)
    pred.prediction(model_dic)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0)
    args = parser.parse_args()
    reg_rate = [1E-6, 1E-5, 1E-5, 1E-5]
    lr = [1E-3, 1E-3, 1E-4, 1E-5]
    config = {
                'attribute_mat_size': 5,
                'reg_rate': reg_rate[args.num],
                'lr': lr[args.num],
                'batch_size':10,
                'gpu_num':2,
                'attributes_num':20,
                'epoch': args.epoch,
                'epoch_mod':1,
                'early_stop_limit':2,
            }
    config['initial_filePath'] = '/datastore/liu121/sentidata2/result/coarse_nn/ckpt_reg1e-05_lr0.001_mat5_senti/'
    config['train_data_file_path'] = '/datastore/liu121/sentidata2/data/aic2018/coarse_data/train_coarse_trail.pkl'
    config['test_data_file_path'] = '/datastore/liu121/sentidata2/data/aic2018/coarse_data/dev_coarse_trail.pkl'
    config['report_filePath'] = '/datastore/liu121/sentidata2/prediction/coarse_nn'
    main(config)


