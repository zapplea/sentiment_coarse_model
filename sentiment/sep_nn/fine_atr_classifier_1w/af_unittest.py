import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
else:
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.util.fine.atr_data_generator import DataGenerator
from sentiment.sep_nn.fine_atr_classifier_1w.classifier import AttributeFunction
from sentiment.sep_nn.fine_atr_classifier_1w.classifier import Classifier

import unittest
import tensorflow as tf
import numpy as np
import math

class AttributeFunctionTest(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(AttributeFunctionTest, self).__init__(*args, **kwargs)

        seed = {'lstm_cell_size': 30,
                'word_dim': 300
                }
        self.nn_config = {'attributes_num': 20,
                          'attribute_senti_prototype_num': 4,
                          'normal_senti_prototype_num': 4,  # number of specific sentiment of each attribute
                          'sentiment_dim': seed['lstm_cell_size'],  # dim of a sentiment expression prototype.
                          'attribute_dim': seed['word_dim'],
                          'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                          'words_num': 10,
                          'word_dim': seed['word_dim'],
                          'attribute_loss_theta': 1.0,
                          'sentiment_loss_theta': 1.0,
                          'is_mat': False,
                          'epoch': None,
                          'rps_num': 5,  # number of relative positions
                          'rp_dim': 15,  # dimension of relative position
                          'lr': 0.003,  # learing rate
                          'batch_size': 30,
                          'lstm_cell_size': seed['lstm_cell_size'],
                          'atr_threshold': 0,  # attribute score threshold
                          'reg_rate': 0.03,
                          'senti_pred_threshold': 0,
                          'lookup_table_words_num': 2981402,  # 2074276 for Chinese word embedding
                          'padding_word_index': 1
                          }

        self.dg = DataGenerator(self.nn_config)

        self.af =AttributeFunction(nn_config=self.nn_config)
        self.cf = Classifier(nn_config=self.nn_config, data_generator = self.dg)
        self.graph = tf.Graph()
        self.sess=tf.Session(graph=self.graph)

    # def test_sentence_input(self):
    #     X_data = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['words_num']),dtype='float32')
    #     with self.graph.as_default():
    #         X = self.cf.sentences_input(self.graph)
    #
    #     result = self.sess.run(X,feed_dict={X:X_data})
    #     self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['words_num']))
    #     test_data = X_data
    #     self.assertTrue(np.all(result == test_data))
    #
    # def test_is_word_padding_input(self):
    #     X = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['words_num']),dtype='float32')*6
    #     X[1][2]=1
    #     with self.graph.as_default():
    #         mask = self.cf.is_word_padding_input(X,self.graph)
    #     result = self.sess.run(mask)
    #
    #     self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['words_num'],self.nn_config['word_dim']))
    #     test_data = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['words_num'],self.nn_config['word_dim']),
    #                         dtype='float32')
    #     test_data[1][2] = np.zeros(shape=(self.nn_config['word_dim'],),dtype='float32')
    #     self.assertTrue(np.all(test_data == result))

    # def test_lookup_table(self):
    #     # data
    #     table_in = np.ones(shape=(2981402, self.nn_config['word_dim']), dtype='float32')
    #     table_in[10000] = np.ones(shape=(self.nn_config['word_dim'],), dtype='float32') * 6
    #     table_in[1] = np.zeros(shape=(self.nn_config['word_dim'],), dtype='float32')
    #     X_data = np.ones(shape=(self.nn_config['batch_size'], self.nn_config['words_num']), dtype='int32') * 7
    #     X_data[9][5] = 10000
    #     X_data[1][4] = 1
    #     X_data[5][6] = 1
    #     # words mask
    #     with self.graph.as_default():
    #         mask = self.cf.is_word_padding_input(X_data, self.graph)
    #     # lookup
    #     with self.graph.as_default():
    #         embedding = self.cf.lookup_table(X_data, mask, self.graph)
    #         table = self.graph.get_collection('table')[0]
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init, feed_dict={table: table_in})
    #         result = sess.run(embedding)
    #
    #     # shape
    #     self.assertEqual(result.shape,
    #                      (self.nn_config['batch_size'], self.nn_config['words_num'], self.nn_config['word_dim']))
    #
    #     # value
    #     self.assertTrue(np.all(
    #         result[4] == np.ones(shape=(self.nn_config['words_num'], self.nn_config['word_dim']), dtype='float32')))
    #     self.assertTrue(np.all(result[9][5] == np.ones(shape=(self.nn_config['word_dim'],), dtype='float32') * 6))
    #     self.assertTrue(np.all(result[5][6] == np.zeros(shape=(self.nn_config['word_dim'],), dtype='float32')))

    # def test_sentence_lstm(self):
    #     # data
    #     table_in = np.ones(shape=(2981402, self.nn_config['word_dim']), dtype='float32')
    #     table_in[10000] = np.ones(shape=(self.nn_config['word_dim'],), dtype='float32') * 6
    #     table_in[1] = np.zeros(shape=(self.nn_config['word_dim'],), dtype='float32')
    #     X_data = np.ones(shape=(self.nn_config['batch_size'], self.nn_config['words_num']), dtype='int32') * 7
    #     X_data[9][5] = 10000
    #     X_data[1][4] = 1
    #     X_data[5][6] = 1
    #
    #     with self.graph.as_default():
    #         mask = self.cf.lstm_mask(X_data)
    #         words_pad_M = self.cf.is_word_padding_input(X_data, self.graph)
    #         X = self.cf.lookup_table(X_data, words_pad_M, self.graph)
    #         table = self.graph.get_collection('table')[0]
    #         outputs = self.cf.sentence_lstm(X,mask,graph=self.graph)
    #         init=tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init,feed_dict={table:table_in})
    #         result = sess.run(outputs)
    #     self.assertEqual(np.array(result).shape,
    #                      (self.nn_config['batch_size'],self.nn_config['words_num'],self.nn_config['lstm_cell_size']))

    # def test_attribute_labels_input(self):
    #     Y_atr_data = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['attributes_num']),dtype='float32')
    #     with self.graph.as_default():
    #         Y_att = self.cf.attribute_labels_input(self.graph)
    #     with self.sess as sess:
    #         result = sess.run(Y_att,feed_dict={Y_att:Y_atr_data})
    #     self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['attributes_num']))
    #     self.assertTrue(np.all(result == Y_atr_data))

    # def test_attribute_vec(self):
    #     with self.graph.as_default():
    #         A,o = self.af.attribute_vec(self.graph)
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result1,result2 = sess.run([A,o])
    #         result3 = sess.run(A-o)
    #
    #     self.assertEqual(result1.shape, (self.nn_config['attributes_num'],self.nn_config['attribute_dim']))
    #     self.assertEqual(result2.shape, (1,self.nn_config['attribute_dim']))
    #     self.assertEqual(result3.shape,(self.nn_config['attributes_num'],self.nn_config['attribute_dim']))

    # def test_attribute_mat(self):
    #     with self.graph.as_default():
    #         A_mat,o_mat = self.af.attribute_mat(self.graph)
    #         A = A_mat-o_mat
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result1,result2 = sess.run([A_mat,o_mat])
    #         result3 = sess.run(A)
    #     self.assertEqual(result1.shape,(self.nn_config['attributes_num'],self.nn_config['attribute_mat_size'],self.nn_config['attribute_dim']))
    #     self.assertEqual(result2.shape,(1,self.nn_config['attribute_mat_size'],self.nn_config['attribute_dim']))
    #     self.assertEqual(result3.shape,(self.nn_config['attributes_num'],self.nn_config['attribute_mat_size'],self.nn_config['attribute_dim']))

    # def test_words_attribute_mat2vec(self):
    #     H_data = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['words_num'],self.nn_config['word_dim']),dtype='float32')
    #     A_mat_data = np.ones(shape=(self.nn_config['attributes_num'], self.nn_config['attribute_mat_size'], self.nn_config['attribute_dim']),
    #                          dtype='float32')
    #
    #     with self.graph.as_default():
    #         words_A = self.af.words_attribute_mat2vec(H_data, A_mat_data,self.graph)
    #     with self.sess as sess:
    #         result = sess.run(words_A)
    #     self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['words_num'],
    #                                    self.nn_config['attributes_num'],self.nn_config['attribute_dim']))
    #     test_data = np.ones_like(result,dtype='float32')
    #     self.assertTrue(np.all(test_data -result < 0.01))

    # def test_words_nonattribute_mat2vec(self):
    #     H_data = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['words_num'],self.nn_config['word_dim']),dtype='float32')
    #     o_mat_data = np.ones(shape=(1,self.nn_config['attribute_mat_size'],self.nn_config['attribute_dim']),dtype='float32')
    #     with self.graph.as_default():
    #         words_o = self.af.words_nonattribute_mat2vec(H_data,o_mat_data,self.graph)
    #     result = self.sess.run(words_o)
    #     self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['words_num'],
    #                                    self.nn_config['attributes_num'],self.nn_config['attribute_dim']))
    #     test_data = np.ones_like(result,dtype='float32')
    #     self.assertTrue(np.all(test_data-result<0.01))

    # def test_A_mat_minus_o_mat(self):
    #     H_data = np.ones(
    #         shape=(self.nn_config['batch_size'], self.nn_config['words_num'], self.nn_config['word_dim']),
    #         dtype='float32')
    #     A_mat_data = np.ones(shape=(self.nn_config['attributes_num'], self.nn_config['attribute_mat_size'], self.nn_config['attribute_dim']),
    #                          dtype='float32')
    #     o_mat_data = np.ones(shape=(1, self.nn_config['attribute_mat_size'], self.nn_config['attribute_dim']),dtype='float32')
    #     with self.graph.as_default():
    #         words_A = self.af.words_attribute_mat2vec(H_data, A_mat_data,self.graph)
    #         words_o = self.af.words_nonattribute_mat2vec(H_data, o_mat_data, self.graph)
    #         A = words_A-words_o
    #     result = self.sess.run(A)
    #     self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['words_num'],
    #                                    self.nn_config['attributes_num'],self.nn_config['attribute_dim']))
    #
    #     test_data = np.zeros_like(result,dtype='float32')
    #     self.assertTrue(np.all(test_data - result == 0))

    # def test_score(self):
    #     self.nn_config['is_mat'] = False
    #     A_data = np.ones(shape=(self.nn_config['attributes_num'],self.nn_config['attribute_dim']),dtype='float32')
    #     X_data = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['words_num'],self.nn_config['word_dim']),dtype='float32')
    #     with self.graph.as_default():
    #         score = self.af.score(A_data,X_data,self.graph)
    #     result = self.sess.run(score)
    #     self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['attributes_num']))
    #     test_data = np.ones_like(score,dtype='float32')*300
    #     self.assertTrue(np.all(test_data == result))
    #
    #
    #     self.nn_config['is_mat'] = True
    #     A_data = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['words_num'],self.nn_config['attributes_num'],self.nn_config['attribute_dim']),
    #                      dtype='float32')
    #     X_data = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['words_num'],self.nn_config['word_dim']),dtype='float32')
    #     with self.graph.as_default():
    #         score = self.af.score(A_data,X_data,self.graph)
    #     result = self.sess.run(score)
    #     self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['attributes_num']))
    #     test_data=np.ones_like(result,dtype='float32')*300
    #     self.assertTrue(np.all(test_data == result))

    # def test_max_false_score(self):
    #     score = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['attributes_num']),dtype='float32')
    #     for i in range(self.nn_config['batch_size']):
    #         for j in range(self.nn_config['attributes_num']):
    #             score[i][j] = j
    #
    #     Y_att = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['attributes_num']),dtype='float32')
    #     for i in range(0,int(math.ceil(self.nn_config['batch_size']/3))):
    #         Y_att[i,0:int(math.ceil(self.nn_config['attributes_num']/3))]=0
    #     for i in range(int(math.ceil(self.nn_config['batch_size']/3)),int(math.ceil(self.nn_config['batch_size']*2/3))):
    #         Y_att[i,int(math.ceil(self.nn_config['attributes_num']/3)):int(math.ceil(self.nn_config['attributes_num']*2/3))] = 0
    #     for i in range(int(math.ceil(self.nn_config['batch_size']*2/3)),self.nn_config['batch_size']):
    #         Y_att[i,int(math.ceil(self.nn_config['attributes_num']*2/3)):self.nn_config['attributes_num']] = 0
    #
    #     with self.graph.as_default():
    #         max_fscore = self.af.max_false_score(score, Y_att, self.graph)
    #     result = self.sess.run(max_fscore)
    #     self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['attributes_num']))
    #
    #     test_data = np.ones_like(result,dtype='float32')
    #     for i in range(0,int(math.ceil(self.nn_config['batch_size']/3))):
    #         test_data[i] = int(math.ceil(self.nn_config['attributes_num']/3))-1
    #     for i in range(int(math.ceil(self.nn_config['batch_size']/3)),int(math.ceil(self.nn_config['batch_size']*2/3))):
    #         test_data[i] = int(math.ceil(self.nn_config['attributes_num']*2/3))-1
    #     for i in range(int(math.ceil(self.nn_config['batch_size']*2/3)),self.nn_config['batch_size']):
    #         test_data[i] = self.nn_config['attributes_num']-1
    #     self.assertTrue(np.all(test_data == result))

    def test_loss(self):
        score = np.ones(shape=(self.nn_config['batch_size'], self.nn_config['attributes_num']), dtype='float32')
        for i in range(self.nn_config['batch_size']):
            for j in range(self.nn_config['attributes_num']):
                score[i][j] = j

        Y_att = np.ones(shape=(self.nn_config['batch_size'], self.nn_config['attributes_num']), dtype='float32')
        for i in range(0, int(math.ceil(self.nn_config['batch_size'] / 3))):
            Y_att[i, 0:int(math.ceil(self.nn_config['attributes_num'] / 3))] = 0
        for i in range(int(math.ceil(self.nn_config['batch_size'] / 3)),
                       int(math.ceil(self.nn_config['batch_size'] * 2 / 3))):
            Y_att[i, int(math.ceil(self.nn_config['attributes_num'] / 3)):int(
                math.ceil(self.nn_config['attributes_num'] * 2 / 3))] = 0
        for i in range(int(math.ceil(self.nn_config['batch_size'] * 2 / 3)), self.nn_config['batch_size']):
            Y_att[i, int(math.ceil(self.nn_config['attributes_num'] * 2 / 3)):self.nn_config['attributes_num']] = 0

        with self.graph.as_default():
            max_fscore = self.af.max_false_score(score,Y_att,self.graph)
            loss = self.af.loss(score,max_fscore,Y_att,self.graph)
        result = self.sess.run(loss)

        self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['attributes_num']))


if __name__ == "__main__":
    unittest.main()