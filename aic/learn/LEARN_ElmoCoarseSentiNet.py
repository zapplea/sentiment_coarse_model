import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import argparse

from aic.coarse_net.senti_net_with_elmo import SentimentNet as CoarseSentiNet
from aic.trains.coarse_senti_train_with_elmo import CoarseSentiTrain
from aic.data_process.senti_datafeeder import DataFeeder

def main(config):
    print('data feeder: ')
    datafeeder = DataFeeder(config)
    print('coarse senti train: ')
    train = CoarseSentiTrain(config, datafeeder)
    print('initial data: ')
    init_data = train.transfer()
    for key in init_data:
        print(key)
    print('========================')
    model_dic = CoarseSentiNet.build(config)
    train.train(model_dic,init_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=3)
    args = parser.parse_args()

    reg_rate = [1E-6, 1E-5, 1E-5, 1E-5]
    lr =       [1E-3, 1E-3, 1E-4, 1E-5]
    config = {
        'attribute_mat_size': 5,
        'reg_rate': reg_rate[args.num],
        'lr': lr[args.num],
        'batch_size': 4,
        'gpu_num': 2,
        'attributes_num': 20,
        'epoch': args.epoch,
        'epoch_mod': 1,
        'early_stop_limit': 2,
        'with_elmo':True,

        'lstm_cell_size': 600,
        'word_dim': 300,
        'attribute_dim': 600,
        'sentiment_dim': 600,
    }
    # TODO: the problem is that: the attribute mat size and the X from lm.weight_layers have different dims.
    # TODO: Just need to adjuct the parameters.
    config['train_data_file_path'] = '/datastore/liu121/sentidata2/data/aic2018/coarse_data/train_coarse.pkl'
    config['test_data_file_path'] = '/datastore/liu121/sentidata2/data/aic2018/coarse_data/dev_coarse.pkl'

    config['attr_sr_path'] = '/datastore/liu121/sentidata2/result/elmo_nn/ckpt_reg%s_lr%s_mat%s_attr/' \
                             % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))
    config['senti_sr_path'] = '/datastore/liu121/sentidata2/result/elmo_nn/ckpt_reg%s_lr%s_mat%s_senti/' \
                              % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))

    config['attr_initial_path'] = config['attr_sr_path']
    config['elmo_initial_path'] = '/datastore/liu121/sentidata2/data/aic2018/elmo_weights/elmo_weights.pkl'

    config['is_restore'] = False
    config['report_filePath'] = '/datastore/liu121/sentidata2/report/elmo_nn/'
    main(config)