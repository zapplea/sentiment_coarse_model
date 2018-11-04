import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import argparse

from aic.coarse_net.attr_net import AttributeNet as CoarseAttributeNet
from aic.fine_net.attr_net import AttributeNet as FineAttributeNet
from aic.trains.transfer_atr_train import TransAtrTrain
from aic.data_process.attr_datafeeder import DataFeeder

def main(config):
    datafeeder = DataFeeder(config)
    train = TransAtrTrain(config, datafeeder)
    fine_net = FineAttributeNet(config)
    init_data = train.transfer(fine_net.classifier)
    coarse_net = CoarseAttributeNet(config)
    train.train(coarse_net.classifier,init_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0)
    args = parser.parse_args()

    reg_rate = [1E-5, ]
    lr =       [1E-4, ]
    config = {
                'attribute_mat_size': 5,
                'reg_rate': reg_rate[args.num],
                'lr': lr[args.num],
                'batch_size':20,
            }
    config['train_data_fine_path'] = '/datastore/liu121/sentidata2/data/aic2018/coarse_data/train_fine.pkl'
    config['test_data_fine_path'] = '/datastore/liu121/sentidata2/data/aic2018/coarse_data/dev_fine.pkl'

    config['sr_path'] = '/datastore/liu121/sentidata2/result/transfer_nn/ckpt_reg%s_lr%s_mat%s' \
                        % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))
    config['initial_path'] = '/datastore/liu121/sentidata2/result/transfer_nn/ckpt_reg%s_lr%s_mat%s'%(',','')
    config['report_filePath'] = '/datastore/liu121/sentidata2/report/fine_nn/report_reg%s_lr%s_mat%s.info' \
                                % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))

    main(config)