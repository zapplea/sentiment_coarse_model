import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" ## 0
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
from sentiment.smartInit_nn.fine_atr_1pNw_smartInit.classifier import Classifier
from sentiment.util.smartInit.namelist_atr_data_generator import DataGenerator

def main(nn_config,data_config):
    dg = DataGenerator(data_config,nn_config)
    cl = Classifier(nn_config, dg)
    cl.train()

if __name__ == "__main__":
    seed = {'lstm_cell_size': 300,
            'word_dim': 300
            }
    nn_configs = [
        {  # fixed parameter
            'attributes_num': 12,
            'attribute_dim': seed['word_dim'],
            'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
            'words_num': 40,
            'word_dim': seed['word_dim'],
            'is_mat': True,
            'epoch': 1000,
            'batch_size': 30,
            'lstm_cell_size': seed['lstm_cell_size'],
            'lookup_table_words_num': 3646,  # 2074276 for Chinese word embedding
            'padding_word_index': 3645,  # the index of #PAD# in word embeddings list
            # flexible parameter
            'reg_rate': 0.00003,
            'lr': 0.0003,  # learing rate
            'atr_pred_threshold': 0,
            # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
            'attribute_loss_theta': 4.0,
        },
        {  # fixed parameter
            'attributes_num': 12,
            'attribute_dim': seed['word_dim'],
            'attribute_mat_size': 2,  # number of attribute mention prototypes in a attribute matrix
            'words_num': 40,
            'word_dim': seed['word_dim'],
            'is_mat': True,
            'epoch': 20000,
            'batch_size': 30,
            'lstm_cell_size': seed['lstm_cell_size'],
            'lookup_table_words_num': 3149,  # 2074276 for Chinese word embedding
            'padding_word_index': 3148,  # the index of #PAD# in word embeddings list
            # flexible parameter
            'reg_rate': 0.001,
            'lr': 0.003,  # learing rate
            'atr_pred_threshold': 0,
            # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
            'attribute_loss_theta': 1.0,
        },
        {  # fixed parameter
            'attributes_num': 12,
            'attribute_dim': seed['word_dim'],
            'attribute_mat_size': 2,  # number of attribute mention prototypes in a attribute matrix
            'words_num': 40,
            'word_dim': seed['word_dim'],
            'is_mat': True,
            'epoch': 10,
            'batch_size': 30,
            'lstm_cell_size': seed['lstm_cell_size'],
            'lookup_table_words_num': 3148,  # 2074276 for Chinese word embedding
            'padding_word_index': 3147,  # the index of #PAD# in word embeddings list
            # flexible parameter
            'reg_rate': 0.0003,
            'lr': 0.0003,  # learing rate
            'atr_pred_threshold': 0,
            # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
            'attribute_loss_theta': 1.0,
        }
    ]
    data_config = {
        'train_source_file_path': '~/dataset/semeval2016/absa_resturant_train.csv',
        'train_data_file_path': '/home/lujunyu/repository/sentiment_coarse_model/sentiment/smartInit_nn/train_data.pkl',
        'test_source_file_path': '~/dataset/semeval2016/absa_resturant_goldentest.csv',
        'test_data_file_path': '/home/lujunyu/repository/sentiment_coarse_model/sentiment/smartInit_nn/test_data.pkl',
        'wordembedding_file_path': '~/dataset/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin',
        'namelist_path':'/home/lujunyu/dataset/semeval2016/namelist/pmi_idf/',
        'stopwords_file_path': '~/dataset/semeval2016/stopwords.txt',
        'attributes_num': nn_configs[0]['attributes_num'],
        'batch_size': nn_configs[0]['batch_size'],
        'words_num': nn_configs[0]['words_num'],
        'padding_word_index': nn_configs[0]['padding_word_index'],
        'word_dim': seed['word_dim']
    }
    # for nn_config in nn_configs:
    #     print(nn_config)
    #     main(nn_config,data_config)
    main(nn_configs[0], data_config)
