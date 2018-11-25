import getpass
import sys
import os
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  ## 0
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
elif getpass.getuser() == 'root':
    sys.path.append('/root/sentiment_coarse_model')

import argparse
from aic.coarse_net.senti_net import SentimentNet
from aic.trains.coarse_senti_train import CoarseSentiTrain
from aic.data_process.senti_datafeeder import DataFeeder

def main(config):
    model_dic = SentimentNet.build(config)
    datafeeder = DataFeeder(config)
    train = CoarseSentiTrain(config,datafeeder)
    train.train(model_dic)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0)
    args = parser.parse_args()

    reg_rate = [1E-5, 1E-5, 1E-5, 1E-5]
    lr =       [1E-4, 1E-3, 1E-2, 1E-1]
    config = {
                'attribute_mat_size': 5,
                'reg_rate': reg_rate[args.num],
                'lr': lr[args.num],
                'batch_size':10,
                'gpu_num':2,
                'attributes_num':20,
                'epoch': 5,
                'epoch_mod':1,
                'early_stop_limit':2
            }

    config['train_data_file_path'] = '/datastore/liu121/sentidata2/data/aic2018/coarse_data/train_coarse.pkl'
    config['test_data_file_path'] = '/datastore/liu121/sentidata2/data/aic2018/coarse_data/dev_coarse.pkl'

    config['sr_path'] = '/datastore/liu121/sentidata2/result/coarse_nn/ckpt_reg%s_lr%s_mat%s/' \
                        % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))
    config['report_filePath'] = '/datastore/liu121/sentidata2/report/coarse_nn/'
    main(config)

