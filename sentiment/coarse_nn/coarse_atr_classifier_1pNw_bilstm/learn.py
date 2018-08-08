import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7" ## 0
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.coarse_nn.coarse_atr_classifier_1pNw_bilstm.classifier import Classifier
from sentiment.util.coarse.atr_data_generator import DataGenerator
from sentiment.util.configure import config

def main(configs):
    dg = DataGenerator(configs)
    cl = Classifier(configs, dg)
    cl.train()


if __name__ == "__main__":
    seed = {'lstm_cell_size': 300,
            'word_dim': 300
            }
    configs = \
        {  # fixed parameter
            'attributes_num': 7,
            'attribute_mat_size': 5,  # number of attribute mention prototypes in a attribute matrix
            'batch_size': 20,
            'max_review_length': 30,
            # flexible parameter
            'reg_rate': 0.0003,
            'lr': 0.003,  # learing rate
            # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
            'attribute_loss_theta': 2.0,
            'aspect_prob_threshold':0.18,
            'keep_prob_lstm':0.5,
            'complement':0,
            'model_save_path': 'ckpt_bi_5mention_8.3/coarse_nn.ckpt',
            'sr_path': '/home/lujunyu/repository/sentiment_coarse_model/sentiment/coarse_nn/coarse_atr_classifier_1pNw/ckpt/'
        }

    configs.update(config)
    main(configs)