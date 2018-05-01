import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
else:
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.multifilter_nn.fine_atr_1pNw_multifilter.classifier import Classifier

import unittest
import numpy as np
import tensorflow as tf

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
                            'attributes_num': 12,
                            'attribute_dim': seed['word_dim'],
                            'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                            'words_num': 10,
                            'word_dim': seed['word_dim'],
                            'is_mat': True,
                            'epoch': 10000,
                            'batch_size': 30,
                            'lstm_cell_size': seed['lstm_cell_size'],
                            'lookup_table_words_num': 100,  # 2074276 for Chinese word embedding
                            'padding_word_index': 0,  # the index of #PAD# in word embeddings list
                            # flexible parameter
                            'reg_rate': 0.03,
                            'lr': 0.3,  # learing rate
                            'atr_pred_threshold': 0,
                        # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
                            'attribute_loss_theta': 1.0,
                            'filter_size': [1, 3, 5],
                            'conv_layer_dim': [1000, seed['lstm_cell_size']]
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
        print('is_mat: ', self.nn_config['is_mat'])
        cl = Classifier(self.nn_config,self.dg)
        cl.classifier()
        print('successful')

    # def test_score(self):
    #     cl=Classifier(self.nn_config,self.dg)
    #     multi_kernel_score = graph.get_collection('multi_kernel_score')[0]
    #     # (batch size, max sentence length, filter_size*word dim)
    #     multi_X = graph.get_collection('multi_X')[0]
    #     multi_H = graph.get_collection('multi_H')[0]
    #     multi_filter = graph.get_collection('multi_filter')[0]
    #     conv_H = graph.get_collection('conv_H')
    #
    #     X_data = np.random.randint(low=1, high=100, size=(5, self.nn_config['words_num'])).astype('int32')
    #     Y_data = np.random.randint(low=0, high=12, size=(5, 12)).astype('float32')
    #     table_data = np.random.randn(100, 200).astype('float32')
    #     # input
    #     X = graph.get_collection('X')[0]
    #     Y = graph.get_collection('Y_att')[0]
    #     table = graph.get_collection('table')[0]
    #     with tf.Session(graph=graph) as sess:
    #         init = tf.global_variables_initializer()
    #         sess.run(init, feed_dict={table: table_data})
    #         # result_ks.shape = (batch size, attributes number, words num, filter numbers)
    #         result_conv_H, result_ks = sess.run([conv_H, multi_kernel_score], feed_dict={X: X_data, Y: Y_data})
    #
    #     for H in result_conv_H:
    #         print(H)
    #     for score in result_ks:
    #         print(score)
    #         # print('==================')
    #         # print(result_ks.shape)
    #         # print(result_ks[0][0])
    #     graph,saver=cl.classifier()




if __name__ == "__main__":
    unittest.main()