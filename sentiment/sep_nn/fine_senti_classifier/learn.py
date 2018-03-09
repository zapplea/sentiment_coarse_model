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

    seed = {'lstm_cell_size': 300,
            'word_dim': 200
            }
    nn_configs = [{ # fixed parameter
                   'attributes_num': 20,
                   'sentiment_dim': seed['lstm_cell_size'],  # dim of a sentiment expression prototype.
                   'attribute_dim': seed['lstm_cell_size'],
                   'words_num': 10,
                   'word_dim': seed['word_dim'],
                   'is_mat': True,
                   'epoch': 10000,
                   'batch_size': 30,
                   'lstm_cell_size': seed['lstm_cell_size'],
                   'lookup_table_words_num': 2981402,  # 2074276 for Chinese word embedding
                   'padding_word_index': 1,
                   # flexible parameter
                   'attribute_senti_prototype_num': 4,
                   'normal_senti_prototype_num': 4,  # number of specific sentiment of each attribute
                   'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                   'sentiment_loss_theta': 1.0,
                   'rps_num': 5,  # number of relative positions
                   'rp_dim': 100,  # dimension of relative position
                   'lr': 0.003,  # learing rate
                   'reg_rate': 0.03,
                   'senti_pred_threshold': 0,
                   'report_filePath': '/datastore/liu121/nosqldb2/sentiA/report1.txt'
                 },
                {  # fixed parameter
                    'attributes_num': 20,
                    'sentiment_dim': seed['lstm_cell_size'],  # dim of a sentiment expression prototype.
                    'attribute_dim': seed['lstm_cell_size'],
                    'words_num': 10,
                    'word_dim': seed['word_dim'],
                    'is_mat': True,
                    'epoch': 10000,
                    'batch_size': 30,
                    'lstm_cell_size': seed['lstm_cell_size'],
                    'lookup_table_words_num': 2981402,  # 2074276 for Chinese word embedding
                    'padding_word_index': 1,
                    # flexible parameter
                    'attribute_senti_prototype_num': 4,
                    'normal_senti_prototype_num': 4,  # number of specific sentiment of each attribute
                    'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                    'sentiment_loss_theta': 1.0,
                    'rps_num': 5,  # number of relative positions
                    'rp_dim': 100,  # dimension of relative position
                    'lr': 0.0003,  # learing rate
                    'reg_rate': 0.0003,
                    'senti_pred_threshold': 0,
                    'report_filePath': '/datastore/liu121/nosqldb2/sentiA/report2.txt'
                }]
    data_config = {}
    for nn_config in nn_configs:
        main(nn_config,data_config)