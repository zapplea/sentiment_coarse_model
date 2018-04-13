import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
else:
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.sep_nn.fine_atr_classifier_nw.classifier import Classifier

import unittest

class DataGenerator:
    def __init__(self):
        pass

class Test(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(Test,self).__init__(*args,**kwargs)
        seed = {'lstm_cell_size': 200,
                'word_dim': 200
                }
        self.nn_config = {  # fixed parameter
                            'attributes_num': 13,
                            'attribute_dim': seed['word_dim'],
                            'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                            'words_num': 20,
                            'word_dim': seed['word_dim'],
                            'is_mat': True,
                            'epoch': 10000,
                            'batch_size': 30,
                            'lstm_cell_size': seed['lstm_cell_size'],
                            'lookup_table_words_num': 3000000,  # 2074276 for Chinese word embedding
                            'padding_word_index': 0,  # the index of #PAD# in word embeddings list
                            # flexible parameter
                            'reg_rate': 0.03,
                            'lr': 0.3,  # learing rate
                            'atr_pred_threshold': 0, # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
                            'attribute_loss_theta': 1.0,
                        }
        self.data_config = {
                                'train_source_file_path': '~/dataset/semeval2016/absa_resturant_train.csv',
                                'train_data_file_path': '/home/lujunyu/repository/sentiment_coarse_model/restaurant attribute data train.pkl',
                                'test_source_file_path': '~/dataset/semeval2016/absa_resturant_test.csv',
                                'test_data_file_path': '/home/lujunyu/repository/sentiment_coarse_model/restaurant attribute data test.pkl',
                                'wordembedding_file_path': '~/dataset/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin',
                                'stopwords_file_path': '~/dataset/semeval2016/stopwords.txt',
                                'testset_size': 1000,
                                'batch_size': self.nn_config['batch_size'],
                                'words_num': self.nn_config['words_num'],
                                'padding_word_index': self.nn_config['padding_word_index']
                            }
        self.dg = DataGenerator()

    def test_classifier(self):
        cl = Classifier(self.nn_config,self.dg)
        cl.classifier()
        print('successful')

if __name__ == "__main__":
    unittest.main()