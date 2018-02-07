import sys
sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
import unittest
from classifier import SentiFunction
from classifier import Classifier
import tensorflow as tf
import numpy as np

class SFTest(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(SFTest, self).__init__(*args, **kwargs)
        seed = {'lstm_cell_size': 30,
                'word_dim': 30
                }
        self.nn_config = {'attributes_num': 20,
                          'attribute_senti_prototype_num': 4,
                          'normal_senti_prototype_num': 4,  # number of specific sentiment of each attribute
                          'sentiment_dim': seed['lstm_cell_size'],  # dim of a sentiment expression prototype.
                          'attribute_dim': seed['word_dim'],
                          'attribute_mat_size': 3,  # number of rows in attribute matrix
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
                          'reg_rate': 0.03
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

    def test_sentence_input(self):
        X_data = np.ones(shape=(self.nn_config['batch_size'],self.nn_config['words_num']))
        with self.graph.as_default():
            X = self.cf.sentences_input(self.graph)
        with self.sess as sess:
            result = sess.run(X,feed_dict={X:X_data})
        self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['words_num']))


    def test_is_word_padding_input(self):
        X_data = np.ones(shape=(self.nn_config['batch_size'], self.nn_config['words_num']))
        with self.graph.as_default():
            mask = self.cf.is_word_padding_input(X_data,self.graph)

    # def test_sentiment_matrix(self):
    #     with self.graph.as_default():
    #         W = self.sf.sentiment_matrix(self.graph)
    #         init = tf.global_variables_initializer()
    #     with tf.Session(graph=self.graph) as sess:
    #         sess.run(init)
    #         result = sess.run(W)
    #     self.assertEqual(result.shape,(self.nn_config['normal_senti_prototype_num']*3+self.nn_config['attribute_senti_prototype_num']*self.nn_config['attributes_num'],
    #                                             self.nn_config['sentiment_dim']))
    #
    # def test_sentiment_attention(self):
    #     E = self.cf.sentiment_extract_mat()
    #     W = np.ones(shape=(self.nn_config['normal_senti_prototype_num']*3+self.nn_config['attribute_senti_prototype_num']*self.nn_config['attributes_num'],
    #                                             self.nn_config['sentiment_dim']),dtype='float32')
    #     W = np.multiply(E,W)
    #     M = self.cf.extor_mask_mat()
    #     attentions = []
    #     with self.graph.as_default():
    #         for j in range(3*self.nn_config['attributes_num']):
    #             attention = self.sf.sentiment_attention(self.x, W[j], M[j], self.graph)
    #             attentions.append(attention)
    #         init = tf.global_variables_initializer()
    #     with self.sess:
    #         self.sess.run(init)
    #         result = self.sess.run(attentions)
    #         # print('result.shape',np.array(result).shape)
    #         self.assertEqual(np.array(result).shape,(3*self.nn_config['attributes_num'],
    #                                                  self.nn_config['words_num'],
    #                                                  self.nn_config['normal_senti_prototype_num']*3+
    #                                                  self.nn_config['attribute_senti_prototype_num']*
    #                                                  self.nn_config['attributes_num']))
    #
    # def test_attended_sentiment(self):
    #     E = self.cf.sentiment_extract_mat()
    #     W = np.ones(shape=(self.nn_config['normal_senti_prototype_num'] * 3 +
    #                        self.nn_config['attribute_senti_prototype_num'] *
    #                        self.nn_config['attributes_num'],
    #                        self.nn_config['sentiment_dim']),
    #                 dtype='float32')
    #     W = np.multiply(E, W)
    #     M = self.cf.extor_mask_mat()
    #     with self.graph.as_default():
    #         W_vec = []
    #         for j in range(3*self.nn_config['attributes_num']):
    #             attention = self.sf.sentiment_attention(self.x,W[j],M[j],self.graph)
    #             w = self.sf.attended_sentiment(W[j],attention,self.graph)
    #             W_vec.append(w)
    #         init = tf.global_variables_initializer()
    #     with self.sess:
    #         self.sess.run(init)
    #         result = self.sess.run(W_vec)
    #
    #     self.assertEqual(np.array(result).shape,(3*self.nn_config['attributes_num'],self.nn_config['words_num'],self.nn_config['sentiment_dim']))
    #
    # def test_attribute_distribution(self):
    #     A = self.A_mat
    #     o = self.o_mat
    #     x = self.x
    #     with self.graph.as_default():
    #         words_A_o = self.af.words_attribute_mat2vec(x=x,A_mat=A,o_mat=o,graph=self.graph)
    #         A = []
    #         o = []
    #         for l in range(len(words_A_o)):
    #             A.append(words_A_o[l][0])
    #             o.append(words_A_o[l][1])
    #         A_dist = self.sf.attribute_distribution(A=A, h=x, graph=self.graph)
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result = sess.run(A_dist)
    #     self.assertEqual(result.shape,(self.nn_config['attributes_num'],self.nn_config['words_num']))
    #
    # def test_relative_pos(self):
    #     with self.graph.as_default():
    #         V = self.sf.relative_pos_matrix(self.graph)
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result = sess.run(V)
    #     self.assertEqual(result.shape,(self.nn_config['rps_num'],self.nn_config['rp_dim']))
    #
    # def test_vi(self):
    #     A = self.A_mat
    #     o = self.o_mat
    #     x = self.x
    #     with self.graph.as_default():
    #         words_A_o = self.af.words_attribute_mat2vec(x=x, A_mat=A, o_mat=o, graph=self.graph)
    #         A = []
    #         o = []
    #         for l in range(len(words_A_o)):
    #             A.append(words_A_o[l][0])
    #             o.append(words_A_o[l][1])
    #         A_dist = self.sf.attribute_distribution(A=A, h=x, graph=self.graph)
    #
    #     with self.graph.as_default():
    #         a_dist = A_dist[0]
    #         V = self.sf.relative_pos_matrix(self.graph)
    #         # vis.shape = (number of words, relative position dim)
    #         vis = []
    #         for i in range(self.nn_config['words_num']):
    #             vi = self.sf.vi(i,a_dist=a_dist,V=V,graph=self.graph)
    #             vis.append(vi)
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result = sess.run(vis)
    #     self.assertEqual(np.array(result).shape,(self.nn_config['words_num'],self.nn_config['rp_dim']))
    #
    # def test_Vi(self):
    #     A = self.A_mat
    #     o = self.o_mat
    #     x = self.x
    #     with self.graph.as_default():
    #         words_A_o = self.af.words_attribute_mat2vec(x=x,A_mat=A,o_mat=o,graph=self.graph)
    #         A = []
    #         o = []
    #         for l in range(len(words_A_o)):
    #             A.append(words_A_o[l][0])
    #             o.append(words_A_o[l][1])
    #         A_dist = self.sf.attribute_distribution(A=A, h=x, graph=self.graph)
    #         V = self.sf.relative_pos_matrix(self.graph)
    #         Vi = self.sf.Vi(A_dist,V,self.graph)
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result = sess.run(Vi)
    #     self.assertEqual(np.array(result).shape,(self.nn_config['attributes_num'],self.nn_config['words_num'],self.nn_config['rp_dim']))
    #
    # def test_beta(self):
    #     with self.graph.as_default():
    #         beta = self.sf.beta(self.graph)
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result = sess.run(beta)
    #     self.assertEqual(result.shape,(self.nn_config['rp_dim'],))
    #
    # def test_item2(self):
    #     A = self.A_mat
    #     o = self.o_mat
    #     x = self.x
    #     with self.graph.as_default():
    #         words_A_o = self.af.words_attribute_mat2vec(x=x, A_mat=A, o_mat=o, graph=self.graph)
    #         A = []
    #         o = []
    #         for l in range(len(words_A_o)):
    #             A.append(words_A_o[l][0])
    #             o.append(words_A_o[l][1])
    #         A_dist = self.sf.attribute_distribution(A=A, h=x, graph=self.graph)
    #         V = self.sf.relative_pos_matrix(self.graph)
    #         Vi = self.sf.Vi(A_dist, V, self.graph)
    #         beta = self.sf.beta(self.graph)
    #         # item2.shape = (attribtes number, words number)
    #         item2 = tf.reduce_sum(tf.multiply(Vi,beta),axis=2)
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result = sess.run(item2)
    #     self.assertEqual(result.shape,(self.nn_config['attributes_num'],self.nn_config['words_num']))
    #
    # def test_item1(self):
    #     E = self.cf.sentiment_extract_mat()
    #     W = np.ones(shape=(self.nn_config['normal_senti_prototype_num'] * 3 +
    #                        self.nn_config['attribute_senti_prototype_num'] *
    #                        self.nn_config['attributes_num'],
    #                        self.nn_config['sentiment_dim']),
    #                 dtype='float32')
    #     W = np.multiply(E, W)
    #     M = self.cf.extor_mask_mat()
    #     item1=[]
    #     with self.graph.as_default():
    #         for j in range(3 * self.nn_config['attributes_num']):
    #             attention = self.sf.sentiment_attention(self.x, W[j], M[j], self.graph)
    #             w = self.sf.attended_sentiment(W[j], attention, self.graph)
    #             item1.append(tf.reduce_sum(tf.multiply(w, self.x), axis=1))
    #         init = tf.global_variables_initializer()
    #     with self.sess:
    #         self.sess.run(init)
    #         result = self.sess.run(item1)
    #     self.assertEqual(np.array(result).shape,(3*self.nn_config['attributes_num'],self.nn_config['words_num']))
    #
    # def test_score(self):
    #     E = self.cf.sentiment_extract_mat()
    #     W = np.ones(shape=(self.nn_config['normal_senti_prototype_num'] * 3 +
    #                        self.nn_config['attribute_senti_prototype_num'] *
    #                        self.nn_config['attributes_num'],
    #                        self.nn_config['sentiment_dim']),
    #                 dtype='float32')
    #     W = np.multiply(E, W)
    #     M = self.cf.extor_mask_mat()
    #     item1 = []
    #     with self.graph.as_default():
    #         for j in range(3 * self.nn_config['attributes_num']):
    #             attention = self.sf.sentiment_attention(self.x, W[j], M[j], self.graph)
    #             w = self.sf.attended_sentiment(W[j], attention, self.graph)
    #             item1.append(tf.reduce_sum(tf.multiply(w, self.x), axis=1))
    #
    #     A = self.A_mat
    #     o = self.o_mat
    #     x = self.x
    #     with self.graph.as_default():
    #         words_A_o = self.af.words_attribute_mat2vec(x=x, A_mat=A, o_mat=o, graph=self.graph)
    #         A = []
    #         o = []
    #         for l in range(len(words_A_o)):
    #             A.append(words_A_o[l][0])
    #             o.append(words_A_o[l][1])
    #         A_dist = self.sf.attribute_distribution(A=A, h=x, graph=self.graph)
    #         V = self.sf.relative_pos_matrix(self.graph)
    #         Vi = self.sf.Vi(A_dist, V, self.graph)
    #         beta = self.sf.beta(self.graph)
    #         # item2.shape = (attribtes number, words number)
    #         item2 = tf.reduce_sum(tf.multiply(Vi, beta), axis=2)
    #         score = self.sf.score(item1,item2,self.graph)
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result = sess.run(score)
    #     self.assertEqual(result.shape,(3*self.nn_config['attributes_num'],))
    #
    # def test_max_f_senti_score(self):
    #     # the senti_label is sentiment label for
    #     senti_label = np.random.randint(0,1,size=(self.nn_config['attributes_num'],3)).astype('float32')
    #     E = self.cf.sentiment_extract_mat()
    #     W = np.ones(shape=(self.nn_config['normal_senti_prototype_num'] * 3 +
    #                        self.nn_config['attribute_senti_prototype_num'] *
    #                        self.nn_config['attributes_num'],
    #                        self.nn_config['sentiment_dim']),
    #                 dtype='float32')
    #     W = np.multiply(E, W)
    #     M = self.cf.extor_mask_mat()
    #     item1 = []
    #     with self.graph.as_default():
    #         for j in range(3 * self.nn_config['attributes_num']):
    #             attention = self.sf.sentiment_attention(self.x, W[j], M[j], self.graph)
    #             w = self.sf.attended_sentiment(W[j], attention, self.graph)
    #             item1.append(tf.reduce_sum(tf.multiply(w, self.x), axis=1))
    #
    #     A = self.A_mat
    #     o = self.o_mat
    #     x = self.x
    #     with self.graph.as_default():
    #         words_A_o = self.af.words_attribute_mat2vec(x=x, A_mat=A, o_mat=o, graph=self.graph)
    #         A = []
    #         o = []
    #         for l in range(len(words_A_o)):
    #             A.append(words_A_o[l][0])
    #             o.append(words_A_o[l][1])
    #         A_dist = self.sf.attribute_distribution(A=A, h=x, graph=self.graph)
    #         V = self.sf.relative_pos_matrix(self.graph)
    #         Vi = self.sf.Vi(A_dist, V, self.graph)
    #         beta = self.sf.beta(self.graph)
    #         # item2.shape = (attribtes number, words number)
    #         item2 = tf.reduce_sum(tf.multiply(Vi, beta), axis=2)
    #         score = self.sf.score(item1, item2, self.graph)
    #         score = tf.reshape(score, shape=(self.nn_config['attributes_num'], 3))
    #         max_fscore = self.sf.max_f_senti_score(senti_label=senti_label,score=score,graph=self.graph)
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result = sess.run(max_fscore)
    #     self.assertEqual(result.shape,(self.nn_config['attributes_num'],))
    #
    # def test_loss(self):
    #     # attributes label for one sentence. one-hot
    #     atr_label = np.random.randint(0, 1, size=(self.nn_config['attributes_num'])).astype('float32')
    #     # the senti_label is sentiment label for
    #     senti_label = []
    #     for i in range(atr_label.shape[0]):
    #         if atr_label[i] == 1:
    #             one_index = np.random.randint(0, 2, size=1)
    #             label = np.zeros(shape=(3,),dtype='float32')
    #             label[one_index] = 1
    #             senti_label.append(label)
    #         else:
    #             label = np.zeros(shape=(3,), dtype='float32')
    #             senti_label.append(label)
    #     senti_label = np.array(senti_label,dtype='float32')
    #
    #     E = self.cf.sentiment_extract_mat()
    #     W = np.ones(shape=(self.nn_config['normal_senti_prototype_num'] * 3 +
    #                        self.nn_config['attribute_senti_prototype_num'] *
    #                        self.nn_config['attributes_num'],
    #                        self.nn_config['sentiment_dim']),
    #                 dtype='float32')
    #     W = np.multiply(E, W)
    #     M = self.cf.extor_mask_mat()
    #     item1 = []
    #     with self.graph.as_default():
    #         for j in range(3 * self.nn_config['attributes_num']):
    #             attention = self.sf.sentiment_attention(self.x, W[j], M[j], self.graph)
    #             w = self.sf.attended_sentiment(W[j], attention, self.graph)
    #             item1.append(tf.reduce_sum(tf.multiply(w, self.x), axis=1))
    #
    #     A = self.A_mat
    #     o = self.o_mat
    #     x = self.x
    #     with self.graph.as_default():
    #         words_A_o = self.af.words_attribute_mat2vec(x=x, A_mat=A, o_mat=o, graph=self.graph)
    #         A = []
    #         o = []
    #         for l in range(len(words_A_o)):
    #             A.append(words_A_o[l][0])
    #             o.append(words_A_o[l][1])
    #         A_dist = self.sf.attribute_distribution(A=A, h=x, graph=self.graph)
    #         V = self.sf.relative_pos_matrix(self.graph)
    #         Vi = self.sf.Vi(A_dist, V, self.graph)
    #         beta = self.sf.beta(self.graph)
    #         # item2.shape = (attribtes number, words number)
    #         item2 = tf.reduce_sum(tf.multiply(Vi, beta), axis=2)
    #         score = self.sf.score(item1, item2, self.graph)
    #         loss = self.sf.loss(senti_label=senti_label,score=score,atr_label=atr_label,graph=self.graph)
    #         init = tf.global_variables_initializer()
    #     with self.sess as sess:
    #         sess.run(init)
    #         result = sess.run(loss)
    #     self.assertEqual(result.shape, (self.nn_config['attributes_num'],))


if __name__ == "__main__":
    unittest.main()