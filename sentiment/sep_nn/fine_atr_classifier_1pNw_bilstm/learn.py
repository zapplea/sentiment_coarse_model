import os
import sys
import getpass
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"  ## 0
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.sep_nn.fine_atr_classifier_1pNw_bilstm.classifier import Classifier
from sentiment.util.fine.atr_data_generator import DataGenerator
from sentiment.util.configure import config

def main(config):
    dg = DataGenerator(config)
    cl = Classifier(config, dg)
    cl.train()


if __name__ == "__main__":
    nn_configs ={
        # fixed parameter

        'attribute_mat_size': 7,  # number of attribute mention prototypes in a attribute matrix
        'attributes_num': 12,
        'batch_size': 200,
        # flexible parameter
        'reg_rate': 0.00003,
        'attribute_loss_theta': 3.0,
        'lr': 0.0003,  # learing rate
        'keep_prob_lstm':0.5,
        'top_k_data': -1
        }
    # for nn_config in nn_configs:
    #     print(nn_config)
    #     main(nn_config,data_config)

    config.update(nn_configs)

    main(config)