import sys
sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
import unittest
from classifier import SentiFunction
from classifier import Classifier
import tensorflow as tf
import numpy as np
import math

class SFTest(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(SFTest, self).__init__(*args, **kwargs)
        seed = {'lstm_cell_size': 30,
                'word_dim': 300
                }
        self.nn_config = {'attributes_num': 20,
                          'attribute_senti_prototype_num': 4,
                          'normal_senti_prototype_num': 4,  # number of specific sentiment of each attribute
                          'sentiment_dim': seed['lstm_cell_size'],  # dim of a sentiment expression prototype.
                          'attribute_dim': seed['lstm_cell_size'],
                          'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                          'words_num': 10,
                          'word_dim': seed['word_dim'],
                          'attribute_loss_theta': 1.0,
                          'sentiment_loss_theta': 1.0,
                          'is_mat': True,
                          'epoch': None,
                          'rps_num': 5,  # number of relative positions
                          'rp_dim': 15,  # dimension of relative position
                          'lr': 0.003,  # learing rate
                          'batch_size': 30,
                          'lstm_cell_size': seed['lstm_cell_size'],
                          'atr_threshold': 0,  # attribute score threshold
                          'reg_rate': 0.03,
                          'senti_pred_threshold':0,
                          'lookup_table_words_num':2981402, # 2074276 for Chinese word embedding
                          'padding_word_index':1
                          }
        self.graph=tf.Graph()
        # self.af = AttributeFunction(self.nn_config)
        self.sf=SentiFunction(self.nn_config)
        self.sess=tf.Session(graph=self.graph)
        self.cf = Classifier(self.nn_config)

        # # data
        # #self.x=np.random.uniform(size=(self.nn_config['words_num'],self.nn_config['word_dim'])).astype('float32')
        # self.x=np.ones(shape=(self.nn_config['words_num'],self.nn_config['word_dim']),dtype='float32')
        # # attributes matrix
        # self.A_mat = np.ones(shape = (self.nn_config['attributes_num'],self.nn_config['attribute_mat_size'],self.nn_config['word_dim']),dtype='float32')
        # self.o_mat = np.ones(shape=(self.nn_config['attribute_mat_size'],self.nn_config['word_dim']),dtype='float32')
        # self.atr_labels= np.random.randint(0,1,size=(self.nn_config['batch_size'],self.nn_config['attributes_num'])).astype(dtype='float32')

    # ######################################
    # ######### sentiment function #########
    # ######################################
    # def test_attribute_vec(self):
    #     with self.graph.as_default():
    #         A_vec = self.sf.attribute_vec(self.graph)
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result = sess.run(A_vec)
    #     self.assertEqual(result.shape, (self.nn_config['attributes_num']+1,self.nn_config['sentiment_dim']))
    #
    # def test_attribute_mat(self):
    #     with self.graph.as_default():
    #         A_mat = self.sf.attribute_mat(self.graph)
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result = sess.run(A_mat)
    #     self.assertEqual(result.shape,(self.nn_config['attributes_num']+1,self.nn_config['attribute_mat_size'],self.nn_config['attribute_dim']))

    # ################################
    # ######### classifier ###########
    # ################################
    # def test_sentence_input(self):
    #     X_data = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['words_num']))
    #     with self.graph.as_default():
    #         X = self.cf.sentences_input(self.graph)
    #     with self.sess as sess:
    #         result = sess.run(X,feed_dict={X:X_data})
    #     self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['words_num']))
    #     self.assertTrue(np.all(result == X_data))

    # def test_is_word_padding_input(self):
    #     X_data = np.ones(shape=(self.nn_config['batch_size'], self.nn_config['words_num']),dtype='int32')*3
    #     X_data[2][3]=4
    #     X_data[1][4]=1
    #     X_data[5][6]=1
    #     with self.graph.as_default():
    #         mask = self.cf.is_word_padding_input(X_data,self.graph)
    #     with self.sess as sess:
    #         result = sess.run(mask)
    #     # shape
    #     self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['words_num'],self.nn_config['word_dim']))
    #
    #     # value
    #     self.assertTrue(np.all(result[2][3] == np.ones(shape=(self.nn_config['word_dim'],),dtype='float32')))
    #     self.assertTrue(np.all(result[6][7] == np.ones(shape=(self.nn_config['word_dim'],),dtype='float32')))
    #     self.assertTrue(np.all(result[1][4] == np.zeros(shape=(self.nn_config['word_dim'],),dtype='float32')))
    #     self.assertTrue(np.all(result[5][6] == np.zeros(shape=(self.nn_config['word_dim'],),dtype='float32')))

    # def test_lookup_table(self):
    #     # data
    #     table_in = np.ones(shape=(2981402, self.nn_config['word_dim']),dtype='float32')
    #     table_in[10000] = np.ones(shape=(self.nn_config['word_dim'],),dtype='float32')*6
    #     table_in[1] = np.zeros(shape=(self.nn_config['word_dim'],),dtype='float32')
    #     X_data = np.ones(shape=(self.nn_config['batch_size'], self.nn_config['words_num']), dtype='int32')*7
    #     X_data[9][5] = 10000
    #     X_data[1][4] = 1
    #     X_data[5][6] = 1
    #     # words mask
    #     with self.graph.as_default():
    #         mask = self.cf.is_word_padding_input(X_data,self.graph)
    #     # lookup
    #     with self.graph.as_default():
    #         embedding = self.cf.lookup_table(X_data,mask,self.graph)
    #         table=self.graph.get_collection('table')[0]
    #         init=tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init,feed_dict={table:table_in})
    #         result = sess.run(embedding)
    #
    #     # shape
    #     self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['words_num'],self.nn_config['word_dim']))
    #
    #     # value
    #     self.assertTrue(np.all(result[4] == np.ones(shape=(self.nn_config['words_num'],self.nn_config['word_dim']),dtype='float32')))
    #     self.assertTrue(np.all(result[9][5] == np.ones(shape=(self.nn_config['word_dim'],),dtype='float32')*6))
    #     self.assertTrue(np.all(result[5][6] == np.zeros(shape=(self.nn_config['word_dim'],),dtype='float32')))

    def test_sentence_lstm(self):
        X_data = np.ones(shape=(self.nn_config['batch_size'], self.nn_config['words_num'],self.nn_config['word_dim']),
                         dtype='float32')
        with self.graph.as_default():
            outputs = self.cf.sentence_lstm(X_data,graph=self.graph)
            init=tf.global_variables_initializer()
        with self.sess as sess:
            sess.run(init)
            result = sess.run(outputs)
        self.assertEqual(np.array(result).shape,
                         (self.nn_config['batch_size'],self.nn_config['words_num'],self.nn_config['lstm_cell_size']))

    # def test_attribute_labels_input(self):
    #     atr_labels_data = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['attributes_num']+1),dtype='float32')
    #     with self.graph.as_default():
    #         y_atti = self.cf.attribute_labels_input(self.graph)
    #     with self.sess as sess:
    #         result = sess.run(y_atti,feed_dict={y_atti:atr_labels_data})
    #     self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['attributes_num']+1))
    #     self.assertTrue(np.all(result == atr_labels_data))

    # def test_sentiment_labels_input(self):
    #     data = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['attributes_num']+1,3),dtype='float32')
    #     with self.graph.as_default():
    #         y_senti = self.cf.sentiment_labels_input(self.graph)
    #     with self.sess as sess:
    #         result = sess.run(y_senti,feed_dict={y_senti:data})
    #     self.assertEqual(result.shape, (self.nn_config['batch_size'],self.nn_config['attributes_num']+1,3))
    #     self.assertTrue(np.all(result == data))

    # def test_sentiment_extract_mat(self):
    #     extors = self.cf.sentiment_extract_mat()
    #
    #     self.assertEqual(extors.shape, (3 * self.nn_config['attributes_num']+3,
    #                                     3 * self.nn_config['normal_senti_prototype_num'] +
    #                                     self.nn_config['attributes_num'] * self.nn_config[
    #                                         'attribute_senti_prototype_num'],
    #                                     self.nn_config['sentiment_dim']))
    #     atr_index_start = 0
    #     atr_index_end = 0
    #     for i in range(3*self.nn_config['attributes_num']):
    #             extor = extors[i]
    #
    #             data = np.zeros_like(extor,dtype='float32')
    #
    #             normal_index_start = (i%3)*self.nn_config['normal_senti_prototype_num']
    #             normal_index_end= normal_index_start+self.nn_config['normal_senti_prototype_num']
    #             for j in range(normal_index_start,normal_index_end):
    #                 data[j] = np.ones_like(extor[0],dtype='float32')
    #
    #             if i%3 == 0:
    #                 count = int(i/3)
    #                 atr_index_start = 3*self.nn_config['normal_senti_prototype_num'] + count*self.nn_config['attribute_senti_prototype_num']
    #                 atr_index_end = atr_index_start+self.nn_config['attribute_senti_prototype_num']
    #             for j in range(atr_index_start,atr_index_end):
    #                 data[j] = np.ones_like(extor[0],dtype='float32')
    #             self.assertTrue(np.all(data == extor))
    #
    #     # test non-attribute's sentiment expression extractor
    #     for i in range(3*self.nn_config['attributes_num'],3*self.nn_config['attributes_num']+3):
    #         extor = extors[i]
    #         data=np.zeros_like(extor,dtype='float32')
    #         normal_index_start = (i % 3) * self.nn_config['normal_senti_prototype_num']
    #         normal_index_end = normal_index_start + self.nn_config['normal_senti_prototype_num']
    #         for j in range(normal_index_start, normal_index_end):
    #             data[j] = np.ones_like(extor[0], dtype='float32')
    #         self.assertTrue(np.all(data == extor))

    # def test_extors_mask(self):
    #     extors = self.cf.sentiment_extract_mat()
    #     with self.graph.as_default():
    #         extor_M = self.cf.extors_mask(extors=extors,graph=self.graph)
    #     with self.sess as sess:
    #         result = sess.run(extor_M)
    #     self.assertEqual(result.shape, (3*self.nn_config['attributes_num'] +3,
    #                                     3 * self.nn_config['normal_senti_prototype_num'] +
    #                                     self.nn_config['attributes_num'] * self.nn_config['attribute_senti_prototype_num']))
    #     atr_index_start = 0
    #     atr_index_end = 0
    #     for i in range(3*self.nn_config['attributes_num']):
    #             mask = result[i]
    #             data = np.zeros_like(mask,dtype='float32')
    #             normal_index_start = (i%3)*self.nn_config['normal_senti_prototype_num']
    #             normal_index_end= normal_index_start+self.nn_config['normal_senti_prototype_num']
    #             for j in range(normal_index_start,normal_index_end):
    #                 data[j] = np.ones_like(mask[0],dtype='float32')
    #
    #             if i%3 == 0:
    #                 count = int(i/3)
    #                 atr_index_start = 3*self.nn_config['normal_senti_prototype_num'] + count*self.nn_config['attribute_senti_prototype_num']
    #                 atr_index_end = atr_index_start+self.nn_config['attribute_senti_prototype_num']
    #             for j in range(atr_index_start,atr_index_end):
    #                 data[j] = np.ones_like(mask[0],dtype='float32')
    #             self.assertTrue(np.all(data == mask))
    #
    #     # test non-attribute extor mask
    #     for i in range(3*self.nn_config['attributes_num'],3*self.nn_config['attributes_num']+3):
    #         mask = result[i]
    #         data=np.zeros_like(mask,dtype='float32')
    #         normal_index_start = (i % 3) * self.nn_config['normal_senti_prototype_num']
    #         normal_index_end = normal_index_start + self.nn_config['normal_senti_prototype_num']
    #         for j in range(normal_index_start, normal_index_end):
    #             data[j] = np.ones_like(mask[0], dtype='float32')
    #         self.assertTrue(np.all(data == mask))

    # def test_sentiment_matrix(self):
    #     with self.graph.as_default():
    #         W = self.sf.sentiment_matrix(self.graph)
    #         init = tf.global_variables_initializer()
    #     with tf.Session(graph=self.graph) as sess:
    #         sess.run(init)
    #         result = sess.run(W)
    #     self.assertEqual(result.shape,(self.nn_config['normal_senti_prototype_num']*3+self.nn_config['attribute_senti_prototype_num']*self.nn_config['attributes_num'],
    #                                    self.nn_config['sentiment_dim']))
    #
    # def test_relative_pos_matrix(self):
    #     with self.graph.as_default():
    #         V = self.sf.relative_pos_matrix(self.graph)
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result = sess.run(V)
    #     self.assertEqual(result.shape, (self.nn_config['rps_num'], self.nn_config['rp_dim']))

    # def test_W_mul_extors(self):
    #     """
    #     Test tf.multiply(W,extors_mat)
    #     :return:
    #     """
    #     with self.graph.as_default():
    #         extors_mat = self.cf.senti_extors_mat(self.graph)
    #         W = self.sf.sentiment_matrix(self.graph)
    #         W = tf.multiply(W,extors_mat)
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result = sess.run(W)
    #     self.assertEqual(result.shape,(3*self.nn_config['attributes_num']+3,
    #                                    self.nn_config['normal_senti_prototype_num'] * 3 + self.nn_config['attribute_senti_prototype_num'] * self.nn_config['attributes_num'],
    #                                    self.nn_config['sentiment_dim']))

    # def test_attribute_mat_attention(self):
    #     att_mat = np.ones(shape=(self.nn_config['attribute_mat_size'],self.nn_config['attribute_dim']),dtype='float32')
    #     word_embed = np.ones(shape=(self.nn_config['lstm_cell_size'],),dtype = 'float32')
    #     with self.graph.as_default():
    #         attention = self.sf.attribute_mat_attention(att_mat,word_embed,self.graph)
    #     with self.sess as sess:
    #         result = sess.run(attention)
    #     self.assertEqual(result.shape,(self.nn_config['attribute_mat_size'],self.nn_config['attribute_dim']))
    #
    #     data=np.ones(shape=(self.nn_config['attribute_mat_size'],self.nn_config['attribute_dim']),dtype='float32')*0.333
    #     self.assertTrue(np.all(result-data>0.0003))

    # def test_attribute_mat2vec(self):
    #     word_embed_data = np.ones(shape=(self.nn_config['lstm_cell_size'],),dtype='float32')
    #     A_mat_data = np.ones(shape=(self.nn_config['attributes_num']+1,self.nn_config['attribute_mat_size'],self.nn_config['lstm_cell_size']),
    #                          dtype='float32')
    #     with self.graph.as_default():
    #         A = self.sf.attribute_mat2vec(word_embed=word_embed_data, A_mat=A_mat_data,graph=self.graph)
    #     with self.sess as sess:
    #         result = sess.run(A)
    #
    #     self.assertEqual(np.array(result).shape,(self.nn_config['attributes_num']+1,self.nn_config['lstm_cell_size']))
    #     test_data = np.ones(shape=(self.nn_config['attributes_num']+1,self.nn_config['lstm_cell_size']),dtype='float32')*0.99
    #     self.assertTrue(np.all(np.array(result)-test_data>0.009))

    # def test_words_attribute_mat2vec(self):
    #     H_data = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['words_num'],self.nn_config['lstm_cell_size']),dtype='float32')
    #     A_mat_data = np.ones(shape=(self.nn_config['attributes_num'] + 1, self.nn_config['attribute_mat_size'], self.nn_config['attribute_dim']),
    #                          dtype='float32')
    #
    #     with self.graph.as_default():
    #         words_A = self.sf.words_attribute_mat2vec(H_data, A_mat_data,self.graph)
    #     with self.sess as sess:
    #         result = sess.run(words_A)
    #
    #     self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['words_num'],self.nn_config['attributes_num']+1,self.nn_config['attribute_dim']))
    #
    #     # test_data = np.ones_like(np.array(result),dtype='float32')
    #     # self.assertTrue(np.all(np.array(result)-test_data == 0))

    # def test_sentiment_attention(self):
    #     h_data = np.ones(shape=(self.nn_config['words_num'], self.nn_config['lstm_cell_size']), dtype='float32')
    #     W = np.ones(shape=(self.nn_config['normal_senti_prototype_num'] * 3 +
    #                        self.nn_config['attribute_senti_prototype_num'] *self.nn_config['attributes_num'],
    #                        self.nn_config['sentiment_dim']),
    #                 dtype='float32')
    #     with self.graph.as_default():
    #         extors_mat = self.cf.senti_extors_mat(self.graph)
    #         extors_mask_mat = self.cf.extors_mask(extors=extors_mat, graph=self.graph)
    #         W = tf.multiply(W,extors_mat)
    #     with self.graph.as_default():
    #         attentions = []
    #         for i in range(3*self.nn_config['attributes_num']+3):
    #             attention = self.sf.sentiment_attention(h=h_data, W=W[i], m=extors_mask_mat[i], graph=self.graph)
    #             attentions.append(attention)
    #
    #     with self.sess as sess:
    #         result = np.array(sess.run(attentions))
    #         extors_mask_mat = sess.run(extors_mask_mat)
    #     self.assertEqual(np.array(result).shape,(3*self.nn_config['attributes_num']+3,
    #                                        self.nn_config['words_num'],
    #                                        3 * self.nn_config['normal_senti_prototype_num'] +
    #                                        self.nn_config['attributes_num'] * self.nn_config['attribute_senti_prototype_num']))
    #
    #     test_data = np.tile(np.expand_dims(extors_mask_mat, axis=1), reps=[1, self.nn_config['words_num'], 1])
    #     for i in range(3*self.nn_config['attributes_num']):
    #         for j in range(self.nn_config['words_num']):
    #             test_data[i][j]=test_data[i][j]*0.125
    #
    #     for i in range(3*self.nn_config['attributes_num'],3*self.nn_config['attributes_num']+3):
    #         for j in range(self.nn_config['words_num']):
    #             test_data[i][j] = test_data[i][j]*0.25
    #
    #     self.assertTrue(np.all(test_data == np.array(result)))

    # def test_attended_sentiment(self):
    #     attention = np.ones(shape=(3*self.nn_config['attributes_num']+3,
    #                                self.nn_config['words_num'],
    #                                3 * self.nn_config['normal_senti_prototype_num'] +
    #                                self.nn_config['attributes_num'] * self.nn_config['attribute_senti_prototype_num']),
    #                         dtype='float32')
    #     W = np.ones(shape=(3*self.nn_config['attributes_num']+3,
    #                        3 * self.nn_config['normal_senti_prototype_num'] +
    #                        self.nn_config['attributes_num'] * self.nn_config['attribute_senti_prototype_num'],
    #                        self.nn_config['sentiment_dim']),
    #                 dtype='float32')
    #     W_vec = []
    #     with self.graph.as_default():
    #         for i in range(3*self.nn_config['attributes_num']+3):
    #             w=self.sf.attended_sentiment(W[i],attention[i],self.graph)
    #             W_vec.append(w)
    #
    #     with self.sess as sess:
    #         result = sess.run(W_vec)
    #     result = np.array(result)
    #     self.assertEqual(result.shape,(3*self.nn_config['attributes_num']+3,
    #                                   self.nn_config['words_num'],
    #                                   self.nn_config['sentiment_dim']))
    #
    #     test_data = np.ones_like(result,dtype='float32')*(3 * self.nn_config['normal_senti_prototype_num'] +self.nn_config['attributes_num'] * self.nn_config['attribute_senti_prototype_num'])
    #     self.assertTrue(np.all(test_data == result))

    # def test_item1(self):
    #     W_vec = np.ones(shape=(3*self.nn_config['attributes_num']+3,
    #                            self.nn_config['words_num'],
    #                            self.nn_config['sentiment_dim']),
    #                     dtype='float32')
    #     h = np.ones(shape=(self.nn_config['words_num'],self.nn_config['sentiment_dim']),
    #                 dtype='float32')
    #
    #     item1 = []
    #     with self.graph.as_default():
    #         for i in range(3*self.nn_config['attributes_num']+3):
    #             w=W_vec[i]
    #             item1.append(tf.reduce_sum(tf.multiply(w, h), axis=1))
    #     with self.sess as sess:
    #         result = sess.run(item1)
    #     result = np.array(result)
    #     self.assertEqual(result.shape,(3*self.nn_config['attributes_num']+3, self.nn_config['words_num']))
    #
    #     test_data = np.ones_like(result,dtype='float32')*self.nn_config['sentiment_dim']
    #     self.assertTrue(np.all(test_data == result))

    # def test_attribute_distribution(self):
    #     if not self.nn_config['is_mat']:
    #         A = np.ones(shape=(self.nn_config['attributes_num']+1,self.nn_config['attribute_dim']),
    #                     dtype='float32')
    #     else:
    #         A = np.ones(shape=(self.nn_config['words_num'],self.nn_config['attributes_num']+1,self.nn_config['attribute_dim']),
    #                     dtype='float32')
    #     h = np.ones(shape=(self.nn_config['words_num'], self.nn_config['sentiment_dim']),
    #                 dtype='float32')
    #     with self.graph.as_default():
    #         A_dist = self.sf.attribute_distribution(A=A,h=h,graph=self.graph)
    #
    #     with self.sess as sess:
    #         result = sess.run(A_dist)
    #
    #     self.assertEqual(result.shape,(self.nn_config['attributes_num']+1,self.nn_config['words_num']))
    #
    #     test_data = np.ones_like(result,dtype='float32')*(1/self.nn_config['words_num'])
    #     self.assertTrue(np.all(test_data == result))

    # def test_vi(self):
    #     a_dist = np.ones(shape = (self.nn_config['words_num'],),
    #                      dtype= 'float32')
    #     i = 2
    #     V = np.ones(shape = (self.nn_config['rps_num'],self.nn_config['rp_dim']),dtype='float32')
    #     V[-1] = np.zeros_like(V[-1],dtype='float32')
    #     with self.graph.as_default():
    #         v = self.sf.vi(i,a_dist,V,self.graph)
    #     with self.sess as sess:
    #         result = sess.run(v)
    #     self.assertEqual(result.shape, (self.nn_config['rp_dim'],))
    #     test_data = np.ones_like(result,dtype='float32')*6
    #     print(result)
    #     self.assertTrue(np.all(test_data == result))

    # def test_Vi(self):
    #     A_dist = np.ones(shape=(self.nn_config['attributes_num']+1,self.nn_config['words_num']),
    #                       dtype='float32')
    #     V = np.ones(shape=(self.nn_config['rps_num'], self.nn_config['rp_dim']), dtype='float32')
    #     V[-1] = np.zeros_like(V[-1], dtype='float32')
    #     with self.graph.as_default():
    #         A_vi = self.sf.Vi(A_dist,V,self.graph)
    #     with self.sess as sess:
    #         result = np.array(sess.run(A_vi))
    #     self.assertEqual(result.shape,(self.nn_config['attributes_num']+1,self.nn_config['words_num'],self.nn_config['rp_dim']))
    #
    #     test_data = np.ones(shape=(self.nn_config['words_num'],self.nn_config['rp_dim']),
    #                       dtype='float32')
    #     test_data[0] = test_data[0]*4
    #     test_data[1] = test_data[1]*5
    #     test_data[2] = test_data[2]*6
    #     test_data[3] = test_data[3]*7
    #     test_data[4] = test_data[4] * 7
    #     test_data[5] = test_data[5] * 7
    #     test_data[6] = test_data[6] * 7
    #     test_data[7] = test_data[7] * 6
    #     test_data[8] = test_data[8] * 5
    #     test_data[9] = test_data[9] * 4
    #     test_data = np.tile(np.expand_dims(test_data,axis=0),reps=[self.nn_config['attributes_num']+1,1,1])
    #     self.assertTrue(np.all(test_data == result))

    # def test_item2(self):
    #     A_vi = np.ones(shape=(self.nn_config['attributes_num']+1,
    #                           self.nn_config['words_num'],
    #                           self.nn_config['rp_dim']),
    #                    dtype='float32')
    #     beta = np.ones(shape=(self.nn_config['rp_dim'],),
    #                    dtype='float32')
    #     with self.graph.as_default():
    #         item2 = tf.reduce_sum(tf.multiply(A_vi, beta), axis=2)
    #     with self.sess as sess:
    #         result = sess.run(item2)
    #     self.assertEqual(result.shape, (self.nn_config['attributes_num']+1,
    #                                     self.nn_config['words_num']))

    # def test_score(self):
    #     item1 = np.ones(shape=(3*self.nn_config['attributes_num']+3,self.nn_config['words_num']),
    #                     dtype='float32')
    #     item2 = np.ones(shape=(self.nn_config['attributes_num']+1,self.nn_config['words_num']),
    #                     dtype='float32')
    #     with self.graph.as_default():
    #         score = self.sf.score(item1,item2,self.graph)
    #     with self.sess as sess:
    #         result = sess.run(score)
    #     self.assertEqual(result.shape,(3*self.nn_config['attributes_num']+3,))
    #
    #     test_data = np.ones_like(result,dtype='float32')*2
    #     self.assertTrue(np.all(test_data == result))

    # def test_max_f_senti_score(self):
    #     score = np.ones(shape=(3*self.nn_config['attributes_num']+3,),
    #                     dtype='float32')
    #     score = np.reshape(score,newshape=(self.nn_config['attributes_num']+1,3))
    #     for i in range(score.shape[0]):
    #         score[i] = np.array([1,2,3],dtype='float32')
    #
    #     senti_label = np.zeros(shape=(self.nn_config['attributes_num']+1,3),
    #                            dtype='float32')
    #     for i in range(0,int(math.ceil(senti_label.shape[0]/3))):
    #         senti_label[i] = np.array([1,0,0],dtype='float32')
    #     for i in range(int(math.ceil(senti_label.shape[0]/3)),int(math.ceil(senti_label.shape[0]*2/3))):
    #         senti_label[i] = np.array([0,1,1],dtype='float32')
    #     for i in range(int(math.ceil(senti_label.shape[0]*2/3)),senti_label.shape[0]):
    #         senti_label[i] = np.array([1,1,1],dtype='float32')
    #
    #     with self.graph.as_default():
    #         max_fscore = self.sf.max_f_senti_score(senti_label,score,self.graph)
    #     with self.sess as sess:
    #         result = sess.run(max_fscore)
    #     self.assertEqual(result.shape, (self.nn_config['attributes_num']+1,3))
    #
    #     test_data = np.ones_like(result,dtype='float32')
    #
    #     for i in range(0,int(math.ceil(senti_label.shape[0]/3))):
    #         test_data[i] = np.array([3,3,3],dtype='float32')
    #     for i in range(int(math.ceil(senti_label.shape[0]/3)),int(math.ceil(senti_label.shape[0]*2/3))):
    #         test_data[i] = np.array([1,1,1],dtype='float32')
    #     for i in range(int(math.ceil(senti_label.shape[0]*2/3)),senti_label.shape[0]):
    #         test_data[i] = np.array([0,0,0],dtype='float32')
    #
    #     self.assertTrue(np.all(result == test_data))
    #
    # def test_senti_loss_mask(self):
    #     atr_label = np.zeros(shape=(self.nn_config['attributes_num']+1,),dtype='float32')
    #     for i in range(0,int(math.ceil(atr_label.shape[0]/2))):
    #         atr_label[i] = 1
    #
    #     senti_label = np.zeros(shape=(self.nn_config['attributes_num']+1,3))
    #     for i in range(0,senti_label.shape[0]):
    #         senti_label[i] = [1,0,0]
    #
    #     with self.graph.as_default():
    #         mask = self.sf.senti_loss_mask(atr_label,senti_label)
    #
    #     with self.sess as sess:
    #         result = sess.run(mask)
    #     self.assertEqual(result.shape,(self.nn_config['attributes_num']+1,3))
    #
    #     test_data = np.zeros_like(result,dtype='float32')
    #     for i in range(0,int(math.ceil(atr_label.shape[0]/2))):
    #         test_data[i] = [1,0,0]
    #
    #     self.assertTrue(np.all(test_data == result))

    # def test_loss(self):
    #     # attribute labels
    #     atr_label = np.zeros(shape=(self.nn_config['attributes_num'] + 1,), dtype='float32')
    #     for i in range(0, int(math.ceil(atr_label.shape[0] / 2))):
    #         atr_label[i] = 1
    #     # sentiment labels
    #     senti_label = np.zeros(shape=(self.nn_config['attributes_num'] + 1, 3))
    #     for i in range(0, int(math.ceil(senti_label.shape[0]))):
    #         senti_label[i] = [1, 0, 0]
    #     # score
    #     score = np.ones(shape=(3 * self.nn_config['attributes_num'] + 3,),
    #                     dtype='float32')
    #     score = np.reshape(score, newshape=(self.nn_config['attributes_num'] + 1, 3))
    #     for i in range(score.shape[0]):
    #         score[i] = np.array([1, 2, 3], dtype='float32')
    #
    #     with self.graph.as_default():
    #         loss = self.sf.loss(senti_label,score,atr_label,self.graph)
    #
    #     with self.sess as sess:
    #         result = sess.run(loss)
    #     self.assertEqual(result.shape,())
    #
    #     test_data = np.array(int(math.ceil((self.nn_config['attributes_num']+1)/2))*3,dtype='float32')
    #     self.assertTrue(test_data == result)

    # def test_prediction(self):
    #     # attribute labels
    #     atr_label = np.zeros(shape=(self.nn_config['attributes_num'] + 1,), dtype='float32')
    #     for i in range(0, int(math.ceil(atr_label.shape[0] / 2))):
    #         atr_label[i] = 1
    #     # score
    #     score = np.ones(shape=(3 * self.nn_config['attributes_num'] + 3,),
    #                     dtype='float32')
    #     score = np.reshape(score, newshape=(self.nn_config['attributes_num'] + 1, 3))
    #     for i in range(score.shape[0]):
    #         score[i] = np.array([-1, -0.2, 1], dtype='float32')
    #
    #     with self.graph.as_default():
    #         pred = self.sf.prediction(score,atr_label,self.graph)
    #     with self.sess as sess:
    #         result = sess.run(pred)
    #
    #     self.assertEqual(result.shape,(self.nn_config['attributes_num']+1,3))
    #
    #     test_data = np.zeros_like(result,dtype='float32')
    #     for i in range(0, int(math.ceil(atr_label.shape[0] / 2))):
    #         test_data[i][2] = 1
    #     self.assertTrue(np.all(test_data == result))

    # def test_classifier(self):
    #     graph,saver = self.cf.classifier()


if __name__ == "__main__":
    unittest.main()