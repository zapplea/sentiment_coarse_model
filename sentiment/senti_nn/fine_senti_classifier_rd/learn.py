import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" ## 0

from sentiment.sep_nn.fine_senti_classifier.classifier import Classifier
from sentiment.util.fine.senti_data_generator import DataGenerator
from sentiment.util.fine.senti_data_generator import DataGenerator_random

def main(nn_config,data_config):
    # dg = DataGenerator_random(data_config)
    dg = DataGenerator(data_config)
    cl = Classifier(nn_config,dg)
    cl.train()


if __name__ == "__main__":

    seed = {'lstm_cell_size': 300,
            'word_dim': 300
            }
    nn_configs = [{ # fixed parameter
                   'attributes_num': 12,
                   'sentiment_dim': seed['lstm_cell_size'],  # dim of a sentiment expression prototype.
                   'attribute_dim': seed['lstm_cell_size'],
                   'words_num': 20,
                   'word_dim': seed['word_dim'],
                   'is_mat': True,
                   'epoch': 10000,#10000
                   'batch_size':30,
                   'lstm_cell_size': seed['lstm_cell_size'],
                   'lookup_table_words_num': 3000000,  # 2074276 for Chinese word embedding
                   'padding_word_index': 0,
                   # flexible parameter
                   'attribute_senti_prototype_num': 4,
                   'normal_senti_prototype_num': 4,  # number of specific sentiment of each attribute
                   'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                   'sentiment_loss_theta': 1.0,
                   'rps_num': 5,  # number of relative positions
                   'rp_dim': 100,  # dimension of relative position
                   'lr': 0.003,  # learing rate
                   'reg_rate': 0.3,
                   'senti_pred_threshold': 0.5,
                   'report_filePath': '/datastore/liu121/nosqldb2/sentiA/report1.txt'
                 },
                {  # fixed parameter
                    'attributes_num': 12,
                    'sentiment_dim': seed['lstm_cell_size'],  # dim of a sentiment expression prototype.
                    'attribute_dim': seed['lstm_cell_size'],
                    'words_num': 10,
                    'word_dim': seed['word_dim'],
                    'is_mat': True,
                    'epoch': 100,
                    'batch_size': 30,
                    'lstm_cell_size': seed['lstm_cell_size'],
                    'lookup_table_words_num': 3000000,  # 2074276 for Chinese word embedding
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
    data_config ={
        'source_file_path':'~/dataset/semeval2016/absa_resturant.csv',
        'data_file_path':'/home/lujunyu/repository/sentiment_coarse_model/restaurant_data.pkl',
        'wordembedding_file_path':'~/dataset/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin',
        'stopwords_file_path': '~/dataset/semeval2016/stopwords.txt',
        'testset_size': 1000,
        'batch_size': nn_configs[0]['batch_size'],
        'words_num': nn_configs[0]['words_num'],
        'padding_word_index': nn_configs[0]['padding_word_index']
    }
    main(nn_configs[0],data_config)