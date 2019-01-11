import getpass
import sys
import os
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"  ## 0
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import argparse

from aic.fine_net.senti_net import SentimentNet
from aic.trains.fine_senti_train import FineSentiTrain
from aic.data_process.senti_datafeeder import DataFeeder

def main(config):
    datafeeder = DataFeeder(config)
    model_dic = SentimentNet.build(config)
    train = FineSentiTrain(config,datafeeder)
    train.train(model_dic)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=3)
    args = parser.parse_args()

    reg_rate = [1E-6, 1E-5, 1E-5, 1E-5]
    lr = [1E-3, 1E-3, 1E-4, 1E-5]
    config = {
                'attribute_mat_size': 5,
                'reg_rate': reg_rate[args.num],
                'lr': lr[args.num],
                'gpu_num':1,
                'batch_size':50,
                'epoch': args.epoch,
                'attributes_num':12,
                'epoch_mod':1,
                'early_stop_limit':20,
                'lookup_table_words_num': 5075,
                'padding_word_index': 5074,
            }

    config['train_data_file_path'] = '/datastore/liu121/sentidata2/data/aic2018/fine_data/train_fine.pkl'
    config['test_data_file_path'] = '/datastore/liu121/sentidata2/data/aic2018/fine_data/dev_fine.pkl'

    config['attr_sr_path'] = '/datastore/liu121/sentidata2/result/fine_nn/ckpt_reg%s_lr%s_mat%s_attr/' \
                             % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))
    config['initial_path'] = config['attr_sr_path']
    config['senti_sr_path'] = '/datastore/liu121/sentidata2/result/fine_nn/ckpt_reg%s_lr%s_mat%s_senti/' \
                              % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))
    config['is_restore'] = False
    config['report_filePath'] = '/datastore/liu121/sentidata2/report/fine_nn/'
    main(config)