import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6" ## 0
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.transfer_nn.transfer_atr_1pNw_bilstm.classifier import Classifier
from sentiment.util.coarse.atr_data_generator import DataGenerator as coarse_DataGenerator
from sentiment.util.fine.atr_data_generator import DataGenerator as fine_DataGenerator

def main(coarse_nn_configs, fine_nn_configs, coarse_data_configs, fine_data_configs):
    coarse_dg = coarse_DataGenerator(coarse_data_configs,coarse_nn_configs)
    fine_dg = fine_DataGenerator(fine_data_configs, fine_nn_configs)
    cl = Classifier(coarse_nn_configs, fine_nn_configs, coarse_dg, fine_dg)
    cl.train()


if __name__ == "__main__":
    seed = {'lstm_cell_size': 300,
            'word_dim': 300
            }
    coarse_nn_configs ={  # fixed parameter
            'attributes_num': 6,
            'attribute_dim': seed['word_dim'],
            'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
            'max_review_length':1,
            'words_num': 40,
            'word_dim': seed['word_dim'],
            'is_mat': True,
            'epoch': 1000,
            'batch_size': 34,
            'lstm_cell_size': seed['lstm_cell_size'],
            'lookup_table_words_num': 30342,  # 2074276 for Chinese word embedding
            'padding_word_index': 30341,  # the index of #PAD# in word embeddings list
            'unk_word_index': 30340,
            # flexible parameter
            'reg_rate': 0.003,
            'lr': 0.0003,  # learing rate
            'atr_pred_threshold': 0,
            # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
            'attribute_loss_theta': 3.0,
            'aspect_prob_threshold':0.2,
            'keep_prob_lstm':0.5,
            'complement':0,
            'sr_path': '/home/lujunyu/repository/sentiment_coarse_model/sentiment/coarse_nn/coarse_atr_classifier_1pNw/ckpt1/'
        }
    coarse_data_configs = {
        'train_source_file_path': '/home/lujunyu/dataset/yelp/yelp_lda_trainset.pkl',
        'train_data_file_path': '/home/lujunyu/repository/sentiment_coarse_model/sentiment/transfer_nn/coarse_train_data.pkl',
        'test_source_file_path': '/home/lujunyu/dataset/yelp/yelp_lda_testset.pkl',
        'test_data_file_path': '/home/lujunyu/repository/sentiment_coarse_model/sentiment/transfer_nn/coarse_test_data.pkl',
        'wordembedding_file_path': '~/dataset/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin',
        'stopwords_file_path': '~/dataset/semeval2016/stopwords.txt',
        'attributes_num': coarse_nn_configs['attributes_num'],
        'batch_size': coarse_nn_configs['batch_size'],
        'words_num': coarse_nn_configs['words_num'],
        'padding_word_index': coarse_nn_configs['padding_word_index'],
        'word_dim': seed['word_dim'],
        'fine_sentences_file':'/home/lujunyu/repository/sentiment_coarse_model/sentiment/transfer_nn/fine_sentences_data.pkl',
        'dictionary':'/home/lujunyu/repository/sentiment_coarse_model/sentiment/util/word_dic/data_dictionary.pkl'
    }
    fine_nn_configs = {  # fixed parameter
            'attributes_num': 12,
            'attribute_dim': seed['word_dim'],
            'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
            'words_num': 40,
            'word_dim': seed['word_dim'],
            'is_mat': True,
            'epoch': 1000,
            'batch_size': 30,
            'lstm_cell_size': seed['lstm_cell_size'],
            'lookup_table_words_num': 30342,  # 2074276 for Chinese word embedding
            'padding_word_index': coarse_nn_configs['padding_word_index'],  # the index of #PAD# in word embeddings list
            'unk_word_index': coarse_nn_configs['unk_word_index'],
            # flexible parameter
            'reg_rate': 0.00003,
            'lr': 0.0003,  # learing rate
            'atr_pred_threshold': 0,
            # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
            'attribute_loss_theta': 3.0,
            'keep_prob_lstm': 0.5
        }
    fine_data_configs = {
        'train_source_file_path': '/home/lujunyu/dataset/semeval2016/absa_resturant_train.pkl',
        'train_data_file_path': '/home/lujunyu/repository/sentiment_coarse_model/sentiment/transfer_nn/fine_train_data.pkl',
        'test_source_file_path': '/home/lujunyu/dataset/semeval2016/absa_resturant_test.pkl',
        'test_data_file_path': '/home/lujunyu/repository/sentiment_coarse_model/sentiment/transfer_nn/fine_test_data.pkl',
        'wordembedding_file_path': '~/dataset/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin',
        'stopwords_file_path': '~/dataset/semeval2016/stopwords.txt',
        'attributes_num': fine_nn_configs['attributes_num'],
        'batch_size': fine_nn_configs['batch_size'],
        'words_num': fine_nn_configs['words_num'],
        'padding_word_index': fine_nn_configs['padding_word_index'],
        'word_dim': seed['word_dim'],
        'dictionary': '/home/lujunyu/repository/sentiment_coarse_model/sentiment/util/word_dic/data_dictionary.pkl'
    }
    # for nn_config in nn_configs:
    #     print(nn_config)
    #     main(nn_config,data_config)
    main(coarse_nn_configs, fine_nn_configs, coarse_data_configs, fine_data_configs)