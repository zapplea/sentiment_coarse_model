import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import argparse

from aic.coarse_net.senti_net import SentimentNet as CoarseSentiNet
from aic.trains.coarse_senti_train_with_elmo import CoarseSentiTrain
from aic.elmo.elmo_net import LanguageModel
from aic.data_process.attr_datafeeder import DataFeeder

def main(config):
    datafeeder = DataFeeder(config)
    train = CoarseSentiTrain(config, datafeeder)
    model_dic = LanguageModel(config)
    init_data = train.transfer(model_dic)
    model_dic = CoarseSentiNet.build(config)
    train.train(model_dic,init_data)

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
            }
    config['train_data_file_path'] = '/datastore/liu121/sentidata2/data/aic2018/coarse_data/train_fine.pkl'
    config['test_data_file_path'] = '/datastore/liu121/sentidata2/data/aic2018/coarse_data/dev_fine.pkl'

    config['sr_path'] = '/datastore/liu121/sentidata2/result/transfer_nn/ckpt_reg%s_lr%s_mat%s' \
                        % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))
    config['initial_file_path'] = '/datastore/liu121/sentidata2/result/fine_nn/ckpt_reg%s_lr%s_mat%s' % ('1E-5', '0.0001', '5')
    config['report_filePath'] = '/datastore/liu121/sentidata2/report/transfer_nn/report_reg%s_lr%s_mat%s.info' \
                                % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))

    main(config)