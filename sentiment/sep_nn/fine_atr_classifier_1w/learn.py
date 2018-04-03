import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" ## 0
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
from sentiment.sep_nn.fine_atr_classifier_1w.classifier import Classifier
from sentiment.util.fine.atr_data_generator import DataGenerator

def main(nn_config,data_config):
    dg = DataGenerator(data_config)
    cl = Classifier(nn_config,dg)
    cl.train()


if __name__ == "__main__":

    seed = {'lstm_cell_size': 300,
            'word_dim': 300
            }
    nn_configs = [
                {  # fixed parameter
                    'attributes_num': 13,
                    'attribute_dim': seed['word_dim'],
                    'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                    'words_num': 20,
                    'word_dim': seed['word_dim'],
                    'is_mat': True,
                    'epoch': 10,
                    'batch_size': 30,
                    'lstm_cell_size': seed['lstm_cell_size'],
                    'lookup_table_words_num': 3000000,  # 2074276 for Chinese word embedding
                    'padding_word_index': 0,  # the index of #PAD# in word embeddings list
                    # flexible parameter
                    'reg_rate': 0.03,
                    'lr': 0.3,  # learing rate
                    'atr_pred_threshold': 0, # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
                    'attribute_loss_theta': 1.0,
                },
                {  # fixed parameter
                    'attributes_num': 13,
                    'attribute_dim': seed['word_dim'],
                    'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                    'words_num': 20,
                    'word_dim': seed['word_dim'],
                    'is_mat': True,
                    'epoch': 10,
                    'batch_size': 30,
                    'lstm_cell_size': seed['lstm_cell_size'],
                    'lookup_table_words_num': 30000000,  # 2074276 for Chinese word embedding
                    'padding_word_index': 0,  # the index of #PAD# in word embeddings list
                    # flexible parameter
                    'reg_rate': 0.03,
                    'lr': 0.03,  # learing rate
                    'atr_pred_threshold': 0, # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
                    'attribute_loss_theta': 1.0,
                },
                {  # fixed parameter
                    'attributes_num': 13,
                    'attribute_dim': seed['word_dim'],
                    'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                    'words_num': 20,
                    'word_dim': seed['word_dim'],
                    'is_mat': True,
                    'epoch': 10,
                    'batch_size': 30,
                    'lstm_cell_size': seed['lstm_cell_size'],
                    'lookup_table_words_num': 30000000,  # 2074276 for Chinese word embedding
                    'padding_word_index': 0,  # the index of #PAD# in word embeddings list
                    # flexible parameter
                    'reg_rate': 0.03,
                    'lr': 0.03,  # learing rate
                    'atr_pred_threshold': 0,
                # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
                    'attribute_loss_theta': 1.0,
                }
                  ]
    data_config = {
        'source_file_path': '~/dataset/semeval2016/absa_resturant.csv',
        'data_file_path': '/home/lujunyu/repository/sentiment_coarse_model/restaurant attribute data.pkl',
        'wordembedding_file_path': '~/dataset/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin',
        'stopwords_file_path': '~/dataset/semeval2016/stopwords.txt',
        'testset_size': 1000,
        'batch_size': nn_configs[0]['batch_size'],
        'words_num': nn_configs[0]['words_num'],
        'padding_word_index': nn_configs[0]['padding_word_index']
    }
    for nn_config in nn_configs:
        main(nn_config,data_config)
    # main(nn_configs[0],data_config)