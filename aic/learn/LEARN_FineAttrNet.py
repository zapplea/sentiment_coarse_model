import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import argparse

from aic.fine_net.attr_net import AttributeNet
from aic.trains.fine_atr_train import FineAtrTrain
from aic.data_process.attr_datafeeder import DataFeeder

def main(config):
    net = AttributeNet(config)
    datafeeder = DataFeeder(config)
    train = FineAtrTrain(config,datafeeder)
    train.train(net.classifier)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0)
    args = parser.parse_args()

    reg_rate = [1E-5, 1E-3, 1E-2, 1E-1]
    lr = [1E-4, 1E-4, 1E-4, 1E-4]
    config = {
                'attribute_mat_size': 5,
                'reg_rate': reg_rate[args.num],
                'lr': lr[args.num],
                'batch_size': 10,
                'gpu_num': 2,
                'attributes_num': None,
                'epoch': 51,
                'epoch_mod': 10,
                'early_stop_limit': 5
            }

    config['train_data_file_path'] = '/datastore/liu121/sentidata2/data/aic2018/fine_data/train_fine.pkl'
    config['test_data_file_path'] = '/datastore/liu121/sentidata2/data/aic2018/fine_data/dev_fine.pkl'

    config['sr_path'] = '/datastore/liu121/sentidata2/result/fine_nn/ckpt_reg%s_lr%s_mat%s/' \
                        % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))
    config['report_filePath'] = '/datastore/liu121/sentidata2/report/fine_nn/'

    main(config)