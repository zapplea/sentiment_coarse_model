import sys
sys.path.append('/home/liu121/dlnlp')
import unittest
from classifier import AttributeFunction
from classifier import Classifier
import tensorflow as tf
import numpy as np

class AFTest(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(AFTest, self).__init__(*args, **kwargs)
        seed = {'lstm_cell_size': 200,
                'word_dim': 200
                }
        self.nn_config = {'attributes_num': 12,
                          'attribute_dim': seed['word_dim'],
                          'words_num': 10,
                          'word_dim': seed['word_dim'],
                          'attribute_loss_theta': 1.0,
                          'epoch': None,
                          'lr': 0.003,  # learing rate
                          'lstm_cell_size': seed['lstm_cell_size'],
                          'atr_score_threshold': 0,  # attribute score threshold for prediction
                          'test_data_size':1000,
                          # reviews' hyper-parameter
                          'batch_size': 15,  # review batch size
                          'sentences_num': 10, # number of sentences in a review
                          'label_atr_threshold': 0.3, # attribute threshold used to eliminate low possibility attributes
                          'wordembedding_file_path': ''
                          }
        self.graph=tf.Graph()
        self.af=AttributeFunction(self.nn_config)
        self.cls = Classifier(self.nn_config)
        self.sess=tf.Session(graph=self.graph)

    def test_attribute_vec(self):
        with self.graph.as_default():
            A,o = self.af.attribute_vec(self.graph)
            init = tf.global_variables_initializer()
        with self.sess:
            self.sess.run(init)
            A = self.sess.run(A)
            o = self.sess.run(o)
            self.assertEqual(np.array(A).shape,(self.nn_config['attributes_num'],self.nn_config['attribute_dim']))
            self.assertEqual(o.shape, (self.nn_config['attribute_dim'],))

    def test_shape_score(self):
        x = np.ones(shape=(self.nn_config['words_num'], self.nn_config['word_dim']), dtype='float32')
        with self.graph.as_default():
            A,o = self.af.attribute_vec(self.graph)
            score = self.af.score(A,o,x,self.graph)
            init = tf.global_variables_initializer()
        with tf.Session(graph=self.graph) as sess:
            sess.run(init)
            result = sess.run(score)
        self.assertEqual(result.shape,(self.nn_config['attributes_num'],))

    def test_review_input(self):
        with self.graph.as_default():
            R = self.cls.reviews_input(self.graph)
            # init = tf.global_variables_initializer()
        # sess.run(init)
        result = self.sess.run(R,feed_dict={R:np.random.randint(low=0,
                                                                high=11,
                                                                size=(self.nn_config['batch_size'],
                                                                self.nn_config['sentences_num'],
                                                                self.nn_config['words_num']),
                                                                dtype='int32')})
        self.assertEqual(result.shape, (self.nn_config['batch_size'],self.nn_config['sentences_num'],self.nn_config['words_num']))

    def test_is_sentence_padding_input(self):
        with self.graph.as_default():
            ispad_M =self.cls.is_sentence_padding_input(self.graph)
        result = self.sess.run(ispad_M,feed_dict={ispad_M:np.random.randint(low=0,
                                                                   high=1,
                                                                   size=(self.nn_config['batch_size'],
                                                                         self.nn_config['sentences_num'])).astype('float32')})
        self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['sentences_num']))

    def test_sentence_lstm(self):
        X = np.random.uniform(size=(self.nn_config['sentences_num'],self.nn_config['words_num'],self.nn_config['word_dim'])).astype('float32')
        with self.graph.as_default():
            # shape = (sentences number, max time, lstm cell size)
            H = self.cls.sentence_lstm(X,self.graph)
            # shape = (sentences number, lstm cell size)
            Hl = tf.transpose(H,[1,0,2])[-1]
            init=tf.global_variables_initializer()
        self.sess.run(init)
        result_H,result_Hl = self.sess.run([H,Hl])
        self.assertEqual(np.array(result_H).shape, (self.nn_config['sentences_num'],self.nn_config['words_num'],self.nn_config['lstm_cell_size']))
        self.assertEqual(Hl.shape,(self.nn_config['sentences_num'],self.nn_config['lstm_cell_size']))

    def test_Sr(self):
        A = np.random.uniform(size=(self.nn_config['attributes_num'],self.nn_config['attribute_dim'])).astype('float32')
        Hl = np.random.uniform(size=(self.nn_config['sentences_num'],self.nn_config['lstm_cell_size'])).astype('float32')
        with self.graph.as_default():
            # shape = (sentences number, attributes number)
            relevance_score=self.cls.Sr(Hl,A,self.graph)
            init = tf.global_variables_initializer()
        self.sess.run(init)
        result = self.sess.run(relevance_score)
        self.assertEqual(result.shape,(self.nn_config['sentences_num'],self.nn_config['attributes_num']))

    def test_sentence_weight(self):
        relvance_score = np.random.uniform(size=(self.nn_config['sentences_num'],self.nn_config['attributes_num'])).astype('float32')
        ispad = np.random.uniform(size=(self.nn_config['sentences_num'],)).astype('float32')
        with self.graph.as_default():
            relevance_weight = self.cls.sentence_weight(relvance_score,ispad,self.graph)
        result = self.sess.run(relevance_weight)
        self.assertEqual(result.shape,(self.nn_config['attributes_num'],self.nn_config['sentences_num']))

    def test_score(self):
        x = np.random.uniform(size=(self.nn_config['words_num'],self.nn_config['word_dim'])).astype('float32')
        A = np.random.uniform(size=(self.nn_config['attributes_num'],self.nn_config['attribute_dim'])).astype('float32')
        o = np.random.uniform(size=(self.nn_config['attribute_dim'],))
        with self.graph.as_default():
            atr_score = self.af.score(A,o,x,self.graph)
        result = self.sess.run(atr_score)
        self.assertEqual(result.shape,(self.nn_config['attributes_num'],))

    def test_loss(self):
        atr_score = np.random.uniform(size=(self.nn_config['attributes_num'],)).astype('float32')
        atr_label = np.random.uniform(low=0,high=1,size=(self.nn_config['attributes_num'],)).astype('float32')
        srw = np.random.uniform(0,1,size=(self.nn_config['attributes_num'],)).astype('float32')
        with self.graph.as_default():
            atr_loss = self.af.loss(atr_score, atr_label, srw, self.graph)
        result = self.sess.run(atr_loss)
        self.assertEqual(result.shape,())

    def test_prediction(self):
        sentences_atr_score = np.random.uniform(size=(self.nn_config['batch_size']*self.nn_config['sentences_num'],
                                                      self.nn_config['attributes_num'])).astype('float32')
        sentences_relevance_weight = np.random.uniform(low=0,
                                                       high=1,
                                                       size=(self.nn_config['batch_size'],
                                                             self.nn_config['sentences_num'],
                                                             self.nn_config['attributes_num'])).astype('float32')
        with self.graph.as_default():
            atr_pred = self.af.prediction(sentences_atr_score, sentences_relevance_weight,self.graph)
        result = self.sess.run(atr_pred)
        self.assertEqual(result.shape,(self.nn_config['batch_size'],self.nn_config['attributes_num']))

    def test_classifier(self):
        # data
        R_d = np.random.randint(low=0,high=3000,size=(self.nn_config['batch_size'],
                                                    self.nn_config['sentences_num'],
                                                    self.nn_config['words_num']))
        ispad_M_d = np.random.randint(low=0,high=1,size=(self.nn_config['batch_size'],
                                                       self.nn_config['sentences_num'])).astype('float32')
        atr_labels_d = np.random.uniform(low=0,high=1,size=(self.nn_config['batch_size'],
                                                          self.nn_config['attributes_num'])).astype('float32')
        # computational graph
        graph,saver= self.cls.classifier()
        with graph.as_default():
            R = graph.get_collection('R')[0]
            ispad_M = graph.get_collection('sentences_padding')[0]
            atr_labels = graph.get_collection('y_att')[0]
            init=tf.global_variables_initializer()
        with tf.Session(graph=graph) as sess:
            sess.run(init)
            #result = sess.run(sentences_loss,feed_dict={R:R_d,ispad_M:ispad_M_d,atr_labels:atr_labels_d})
        #self.assertEqual(np.array(result).shape, (self.nn_config['batch_size']*self.nn_config['sentences_num'],))


if __name__=="__main__":
    unittest.main()