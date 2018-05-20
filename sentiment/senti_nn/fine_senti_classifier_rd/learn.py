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
        'attribute_dim': seed['word_dim'],
        'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
        'words_num': 40,
        'word_dim': seed['word_dim'],
        'is_mat': True,
        'epoch': 1000,
        'batch_size': 10,
        'lstm_cell_size': seed['lstm_cell_size'],
        'lookup_table_words_num': 30342,  # 2074276 for Chinese word embedding
        'padding_word_index': 30341,  # the index of #PAD# in word embeddings list
        # flexible parameter
        'reg_rate': 0.003,
        'lr': 0.00001,  # learing rate
        'atr_pred_threshold': 0,
        # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
        'attribute_loss_theta': 3.0,
        'keep_prob_lstm': 0.5,
        'attribute_senti_prototype_num': 4, #10  number of sentiment prototypes for a specific attribute
        'normal_senti_prototype_num': 4, #10 number of sentiment prototypes for each normal sentiment polarity like "Negative"
        'sentiment_dim': seed['lstm_cell_size'], # dim of a sentiment expression prototype.
        'rel_words_num':20, # number of dependency-parser relation words
        'rel_word_dim':seed['word_dim'], # a relation word's dimension
        'sentiment_loss_theta': 1.0, # deprecated
        'rps_num': 5, # number of relative distance. if it is 5, then it means , for word_i, the distance between the other words and word_i is at most 5. if the distance is greater than 5, then we still consider the distance as 5.
        'rp_dim': seed['lstm_cell_size'], # dimension of relative distance embedding
        'atr_threshold': 0, # attribute score threshold, not used in here
        'senti_pred_threshold':0, # deprecated, used to predict sentiment
        'max_path_length':20 # the maximal length of dependency path, the util/dependency_parser/dependency_generator.py will print it out.
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