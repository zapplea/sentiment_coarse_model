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
from aic.fine_net.senti_net import SentimentNet as FineSentiNet
from aic.trains.transfer_senti_train import TransSentiTrain
from aic.data_process.attr_datafeeder import DataFeeder

def main(config):
    datafeeder = DataFeeder(config)
    train = TransSentiTrain(config, datafeeder)
    model_dic = FineSentiNet.build(config)
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
    config['tfb_filePath']='/datastore/liu121/sentidata2/resultdata/fine_nn/model/ckpt_reg%s_lr%s_mat%s' \
                            % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))
    config['report_filePath']='/datastore/liu121/sentidata2/resultdata/fine_nn/report/report_reg%s_lr%s_mat%s.info' \
                              % (str(reg_rate[args.num]), str(lr[args.num]), str(config['attribute_mat_size']))

    main(config)