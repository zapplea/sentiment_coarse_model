import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
else:
    sys.path.append('/home/liu121/sentiment_coarse_model')
from sentiment.coarse_nn.relevance_score.relevance_score import RelScore

import unittest
import numpy as np
import tensorflow as tf


class Test(unittest.TestCase):
    def __init__(self,*arg,**kwargs):
        super(Test,self).__init__(*arg,**kwargs)
        seed = {'lstm_cell_size': 200,
                'word_dim': 200
                }
        self.nn_config = {  # fixed parameter
                            'attributes_num': 12,
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
                            'atr_pred_threshold': 0,
                            # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
                            'attribute_loss_theta': 1.0,
                            'max_review_length': 5,
                            'aspect_prob_threshold': 0.3
                        }
        self.relscore= RelScore(self.nn_config)

    def test_reviews_input(self):
        graph = tf.Graph()
        with graph.as_default():
            X_data = np.ones(shape=(20,self.nn_config['max_review_length'],self.nn_config['words_num']),dtype='int32')
            X = self.relscore.reviews_input(graph)
            X_input = graph.get_collection('X')[0]
            sess = tf.Session()
            result = sess.run(X,feed_dict={X_input:X_data})
            self.assertEqual(result.shape,(20*self.nn_config['max_review_length'],self.nn_config['words_num']))

    def test_aspect_prob2true_label(self):
        data = np.array([[0.3,0.4,0.5,0.6,0.3,0.4,0.5,0.6,0.3,0.4,0.5,0.6],[0.2,0.1,0.4,0.5,0.2,0.1,0.4,0.5,0.2,0.1,0.4,0.5]],)
        graph = tf.Graph()
        with graph.as_default():
            true_label = self.relscore.aspect_prob2true_label(data)
            sess=tf.Session()
            result = sess.run(true_label)
        test_data = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],])
        self.assertTrue(np.all(np.equal(result,test_data)))

    def test_relevance_prob_atr(self):
        atr_score = np.ones(shape=(3*self.nn_config['max_review_length'],self.nn_config['attributes_num']),dtype='float32')
        graph = tf.Graph()
        with graph.as_default():
            rp = self.relscore.relevance_prob_atr(atr_score,graph)
            sess = tf.Session()
            result = sess.run(rp)
        self.assertEqual(result.shape,(3*self.nn_config['max_review_length'],self.nn_config['attributes_num']))
        test_data = np.ones(shape=(3*self.nn_config['max_review_length'],self.nn_config['attributes_num']),dtype='float32')*(1/self.nn_config['max_review_length'])
        self.assertTrue(np.all(np.equal(test_data,result)))

    def test_expand_aspect_prob(self):
        aspect_prob = np.random.uniform(0,1,size=(3,self.nn_config['attributes_num'])).astype('float32')
        graph = tf.Graph()
        with graph.as_default():
            exp_aspect_prob = self.relscore.expand_aspect_prob(aspect_prob,graph)
            sess = tf.Session()
            result = sess.run(exp_aspect_prob)
        self.assertEqual(result.shape,(3*self.nn_config['max_review_length'],self.nn_config['attributes_num']))


if __name__ == "__main__":
    unittest.main()