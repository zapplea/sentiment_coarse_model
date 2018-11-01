import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import argparse

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  ## 0

from aic.fine_net.senti_net import SentimentNet
from aic.trains.fine_senti_train import FineSentiTrain
from aic.data_process.senti_datafeeder import DataFeeder

def main(config):
    model_dic = SentimentNet.build(config)
    datafeeder = DataFeeder(config)
    train = FineSentiTrain(config,datafeeder)
    train.train(model_dic)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0)
    args = parser.parse_args()

    reg_rate = [1E-5, ]
    lr =       [1E-4, ]
    config = {
                'attribute_mat_size': 5,
                'reg_rate': reg_rate[args.num],
                'lr': lr[args.num]
            }
    config['tfb_filePath']='/hdd/lujunyu/model/meituan/coarse_nn/model/sentiment/ckpt_reg%s_lr%s_mat%s' \
                            % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))
    config['report_filePath']='/hdd/lujunyu/model/meituan/coarse_nn/report/sentiment/report_reg%s_lr%s_mat%s.info' \
                              % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))

    main(config)