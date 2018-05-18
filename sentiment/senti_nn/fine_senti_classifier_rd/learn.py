import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" ## 0

from sentiment.senti_nn.fine_senti_classifier_rd.classifier import Classifier
from sentiment.util.fine.senti_data_generator import DataGenerator
# from sentiment.util.fine.senti_data_generator import DataGenerator

def main(nn_config,data_config):
    # dg = DataGenerator_random(data_config)
    dg = DataGenerator(data_config,nn_configs)
    cl = Classifier(nn_config,dg)
    cl.train()


if __name__ == "__main__":

    seed = {'lstm_cell_size': 300,
            'word_dim': 300
            }
    nn_configs = { # fixed parameter
                   'attributes_num': 12,
                   'sentiment_dim': seed['lstm_cell_size'],  # dim of a sentiment expression prototype.
                   'attribute_dim': seed['lstm_cell_size'],
                   'words_num': 40,
                   'word_dim': seed['word_dim'],
                   'is_mat': True,
                   'epoch': 10000,#10000
                   'batch_size':17,
                   'lstm_cell_size': seed['lstm_cell_size'],
                   'lookup_table_words_num': 30342,  # 2074276 for Chinese word embedding
                   'padding_word_index': 30341,
                   # flexible parameter
                   'attribute_senti_prototype_num': 4,
                   'normal_senti_prototype_num': 4,  # number of specific sentiment of each attribute
                   'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                   'sentiment_loss_theta': 3.0,
                   'rps_num': 5,  # number of relative positions
                   'rp_dim': 100,  # dimension of relative position
                   'lr': 0.00003,  # learing rate
                   'reg_rate': 0.003,
                   'senti_pred_threshold': 0.5,
                   'keep_prob_lstm': 0.5,
                   'report_filePath': '/datastore/liu121/nosqldb2/sentiA/report1.txt'
                 }
    data_config = {
        'train_source_file_path': '/home/lujunyu/dataset/semeval2016/absa_resturant_train.pkl',
        'train_data_file_path': '/home/lujunyu/repository/sentiment_coarse_model/sentiment/senti_nn/train_data.pkl',
        'test_source_file_path': '/home/lujunyu/dataset/semeval2016/absa_resturant_test.pkl',
        'test_data_file_path': '/home/lujunyu/repository/sentiment_coarse_model/sentiment/senti_nn/test_data.pkl',
        'wordembedding_file_path': '~/dataset/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin',
        'stopwords_file_path': '~/dataset/semeval2016/stopwords.txt',
        'attributes_num': nn_configs['attributes_num'],
        'batch_size': nn_configs['batch_size'],
        'words_num': nn_configs['words_num'],
        'padding_word_index': nn_configs['padding_word_index'],
        'word_dim': seed['word_dim'],
        'top_k_data': -1,
        'dictionary': '/home/lujunyu/repository/sentiment_coarse_model/sentiment/util/word_dic/data_dictionary.pkl'
    }
    main(nn_configs,data_config)