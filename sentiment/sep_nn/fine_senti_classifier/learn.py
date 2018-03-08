import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
else:
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.sep_nn.fine_senti_classifier.classifier import Classifier
from sentiment.util.fine.senti_data_generator import DataGenerator

def main(nn_config,data_config):
    dg = DataGenerator(data_config)
    cl = Classifier(nn_config,dg)
    cl.train()


if __name__ == "__main__":
    seed = {'lstm_cell_size': 30,
            'word_dim': 300
            }
    nn_config = {'attributes_num': 20,
                 'attribute_senti_prototype_num': 4,
                 'normal_senti_prototype_num': 4,  # number of specific sentiment of each attribute
                 'sentiment_dim': seed['lstm_cell_size'],  # dim of a sentiment expression prototype.
                 'attribute_dim': seed['lstm_cell_size'],
                 'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                 'words_num': 10,
                 'word_dim': seed['word_dim'],
                 'attribute_loss_theta': 1.0,
                 'sentiment_loss_theta': 1.0,
                 'is_mat': True,
                 'epoch': 10000,
                 'rps_num': 5,  # number of relative positions
                 'rp_dim': 15,  # dimension of relative position
                 'lr': 0.003,  # learing rate
                 'batch_size': 30,
                 'lstm_cell_size': seed['lstm_cell_size'],
                 'atr_threshold': 0,  # attribute score threshold
                 'reg_rate': 0.03,
                 'senti_pred_threshold': 0,
                 'lookup_table_words_num': 2981402,  # 2074276 for Chinese word embedding
                 'padding_word_index': 1
                  }
    main(nn_config)