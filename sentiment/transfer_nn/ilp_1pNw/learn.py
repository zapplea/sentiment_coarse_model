import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" ## 0
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model/')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.transfer_nn.ilp_1pNw.classifier import Classifier
from sentiment.util.coarse.atr_data_generator import DataGenerator as coarse_DataGenerator
from sentiment.util.fine.atr_data_generator import DataGenerator as fine_DataGenerator
from sentiment.util.configure import config

def main(coarse_configs, fine_configs):
    coarse_dg = coarse_DataGenerator(coarse_configs)
    fine_dg = fine_DataGenerator(fine_configs)
    cl = Classifier(coarse_configs, fine_configs, coarse_dg, fine_dg)
    cl.train()


if __name__ == "__main__":
    coarse_configs ={  # fixed parameter
        'attributes_num': 6,
        'attribute_mat_size': 5,  # number of attribute mention prototypes in a attribute matrix
        'max_review_length':1,
        'batch_size': 34,
        # flexible parameter
        'reg_rate': 0.003,
        'lr': 0.0003,  # learing rate
        # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
        'attribute_loss_theta': 3.0,
        'aspect_prob_threshold':0.2,
        'keep_prob_lstm':0.5,
        'complement':0,
        'sr_path': '/home/lujunyu/repository/sentiment_coarse_model/sentiment/coarse_nn/coarse_atr_classifier_1pNw_bilstm/ckpt_bi_5mention_7.12/',
        'fine_sentences_file':'/home/lujunyu/repository/sentiment_coarse_model/sentiment/transfer_nn/fine_sentences_data.pkl'
    }

    fine_configs = {  # fixed parameter
        'attributes_num': 12,
        'attribute_mat_size': 5,  # number of attribute mention prototypes in a attribute matrix
        'batch_size': 10,
        # flexible parameter
        'reg_rate': 0.00003,
        'lr': 0.0003,  # learing rate
        'atr_pred_threshold': 0,
        # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
        'attribute_loss_theta': 3.0,
        'keep_prob_lstm': 0.5,
        'top_k_data': 30
    }

    coarse_configs.update(config)
    fine_configs.update(config)


    main(coarse_configs, fine_configs)