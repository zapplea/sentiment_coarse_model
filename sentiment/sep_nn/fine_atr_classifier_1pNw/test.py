import os
import sys

sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.sep_nn.fine_atr_classifier_1pNw.classifier import Classifier

import unittest
import tensorflow as tf
import numpy as np

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
                            'words_num': 5,
                            'word_dim': seed['word_dim'],
                            'is_mat': True,
                            'epoch': 10000,
                            'batch_size': 30,
                            'lstm_cell_size': seed['lstm_cell_size'],
                            'lookup_table_words_num': 100,  # 2074276 for Chinese word embedding
                            'padding_word_index': 2,  # the index of #PAD# in word embeddings list
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
        self.cl = Classifier(self.nn_config, self.dg)
        self.graph=tf.Graph()

    def test_sentence_input(self):
        X_data = np.ones(shape=(5,self.nn_config['words_num']),dtype='float32')
        with self.graph.as_default():
            X = self.cl.sentences_input(self.graph)
            sess = tf.Session()
            result = sess.run(X,feed_dict={X:X_data})
        self.assertTrue(np.all(np.equal(result,X_data)))

    def test_is_word_padding_input(self):
        batch_size = 2
        X_ids= np.ones(shape=(batch_size,self.nn_config['words_num']),dtype='int32')
        for i in range(2):
            for j in [-1,-2,-3]:
                X_ids[i][j]=X_ids[i][j]*self.nn_config['padding_word_index']
        with self.graph.as_default():
            mask = self.cl.is_word_padding_input(X_ids,self.graph)
            sess = tf.Session()
            result = sess.run(mask)
        self.assertEqual(result.shape,(batch_size,self.nn_config['words_num'],self.nn_config['word_dim']))

        test_data = np.array([[1,1,0,0,0],[1,1,0,0,0]],dtype='float32')
        test_data = np.tile(np.expand_dims(test_data,axis=2),[1,1,self.nn_config['word_dim']])
        self.assertTrue(np.all(np.equal(test_data,result)))

    def test_lookup_table(self):
        batch_size = 2
        X_ids = np.ones(shape=(batch_size, self.nn_config['words_num']), dtype='int32')
        for i in range(2):
            for j in [-1,-2,-3]:
                X_ids[i][j]=X_ids[i][j]*self.nn_config['padding_word_index']
        mask = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]], dtype='float32')
        mask = np.tile(np.expand_dims(mask, axis=2), [1, 1, self.nn_config['word_dim']])

        table_data=np.random.uniform(size=(self.nn_config['lookup_table_words_num'],self.nn_config['word_dim'])).astype('float32')
        table_data[self.nn_config['padding_word_index']] = np.zeros(shape=(self.nn_config['word_dim'],),dtype='float32')
        with self.graph.as_default():
            X = self.cl.lookup_table(X_ids,mask,self.graph)
            table = self.graph.get_collection('table')[0]
            sess = tf.Session()
            init=tf.global_variables_initializer()
            sess.run(init,feed_dict={table:table_data})
            result = sess.run(X)
        self.assertEqual(result.shape,(batch_size,self.nn_config['words_num'],self.nn_config['word_dim']))
        print(result)

    def test_sequence_length(self):
        batch_size = 2
        X_ids = np.ones(shape=(batch_size, self.nn_config['words_num']), dtype='int32')
        for i in range(2):
            for j in [-1, -2, -3]:
                X_ids[i][j] = X_ids[i][j] * self.nn_config['padding_word_index']
        with self.graph.as_default():
            seq_len = self.cl.sequence_length(X_ids,self.graph)
            sess= tf.Session()
            result = sess.run(seq_len)
            print(result)

    def test_classifier(self):
        self.cl.classifier()

if __name__ == "__main__":
    unittest.main()