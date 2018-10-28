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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  ## 0

from aic.coarse_net.attr_net import AttributeNet
from aic.trains.coarse_atr_train import CoarseAtrTrain
from aic.data_process.attr_datafeeder import DataFeeder

def main(config):
    net = AttributeNet(config)
    datafeeder = DataFeeder(config)
    train = CoarseAtrTrain(config,datafeeder)
    train.train(net.classifier)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0)
    args = parser.parse_args()

    reg_rate = [1E-5, ]
    lr       = [1E-4, ]
    config = {
                'attribute_mat_size': 5,
                'reg_rate': reg_rate[args.num],
                'lr': lr[args.num],
            }
    config['tfb_filePath']='/hdd/lujunyu/model/meituan/coarse_nn/model/ckpt_reg%s_lr%s_mat%s' \
                            % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))
    config['report_filePath']='/hdd/lujunyu/model/meituan/coarse_nn/report/report_reg%s_lr%s_mat%s.info' \
                              % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))

    main(config)