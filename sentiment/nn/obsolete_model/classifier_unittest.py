import unittest
import classifier_att_lstm
import tensorflow as tf
import numpy as np

class ClassifierTest(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(ClassifierTest, self).__init__(*args, **kwargs)
        init = {'sentence_words_num': 20,
                'word_dim': 200,
                'batch_size': 30,
                'sentence_cell_size': 500,
                'sentence_num': 10,
                'aspect_words_num': 5,
                'aspect_rnn_cell_size': 300
                }
        self.rnn_config = {'shape': {'rnn': (init['word_dim'], init['sentence_cell_size']),
                                'input': (
                                    init['sentence_num'], init['batch_size'], init['sentence_words_num'], init['word_dim']),
                                'aspect': (init['batch_size'], init['aspect_words_num'], init['word_dim']),
                                'gt': (init['batch_size'], 1),
                                'aspect_rnn': (init['word_dim'], init['aspect_rnn_cell_size']),
                                'att_linear_weight': (init['aspect_rnn_cell_size'] + init['word_dim'], 500),
                                'att_softmax_weight': (500, 1)
                                },
                      'name': {'rnn': 'rnn', 'hidden': 'hidden', 'sigmoid': 'sigmoid', 'softmax': 'softmax',
                               'aspect_rnn': 'aspect_rnn',
                               'attention': 'att'},
                      'batch_size': init['batch_size'],
                      'n_steps': init['sentence_words_num'],
                      'sentence_cell_size': init['sentence_cell_size'],
                      'sentence_num': init['sentence_num'],
                      'hidden_layer_dim': [init['sentence_cell_size'], 500],  # for rnn hidden layer
                      'lr': 0.003,
                      'lamda': 0.03,
                      'word_dim': init['word_dim'],
                      'sentence_words_num': init['sentence_words_num'],
                      'aspect_words_num': init['aspect_words_num'],
                      'aspect_rnn_cell_size': init['aspect_rnn_cell_size'],
                      'aspect_rnn_n_steps': init['aspect_words_num'],
                      'epoch': 1000,
                      'sl_path': '/liu121/model'
                      }
        self.graph,_=classifier_att_lstm.Classifier(self.rnn_config).classifier()
        self.data_aspect=np.ones(shape=(self.rnn_config['batch_size'],self.rnn_config['aspect_words_num'],self.rnn_config['word_dim']),dtype='float32')
        self.data_sentence=np.ones(shape=(self.rnn_config['sentence_num'],self.rnn_config['batch_size'],self.rnn_config['sentence_words_num'],self.rnn_config['word_dim']))

    def test_aspect_rnn(self):
        with self.graph.as_default():
            A=self.graph.get_collection('aspect')[0]
            aspect_rnn=self.graph.get_collection('aspect_rnn_output')[0]
            init=tf.global_variables_initializer()

        with tf.Session(graph=self.graph) as sess:
            sess.run(init)
            result_aspect,result_aspect_rnn=sess.run([A,aspect_rnn],feed_dict={A:self.data_aspect})
            self.assertEqual(result_aspect.shape,self.data_aspect.shape)
            self.assertEqual(result_aspect_rnn.shape,(self.rnn_config['batch_size'],self.rnn_config['aspect_rnn_cell_size']))

    def test_aspect_transform(self):
        with self.graph.as_default():
            A = self.graph.get_collection('aspect')[0]
            A_trans = self.graph.get_collection('aspect_transformed')[0]
            init = tf.global_variables_initializer()

        with tf.Session(graph=self.graph) as sess:
            sess.run(init)
            result_A_trans = sess.run(A_trans, feed_dict={A:self.data_aspect})
            self.assertEqual(result_A_trans.shape,(self.rnn_config['sentence_num'],self.rnn_config['batch_size'],1,self.rnn_config['aspect_rnn_cell_size']))

    def test_attention(self):
        with self.graph.as_default():
            A = self.graph.get_collection('aspect')[0]
            X = self.graph.get_collection('input')[0]
            attention=self.graph.get_collection('attention')[0]
            init=tf.global_variables_initializer()
        with tf.Session(graph=self.graph) as sess:
            sess.run(init)
            result_attention=sess.run(attention,feed_dict={X:self.data_sentence,A:self.data_aspect})
            self.assertEqual(result_attention.shape,(self.rnn_config['sentence_num'],self.rnn_config['batch_size'],
                                                     self.rnn_config['sentence_words_num'],self.rnn_config['word_dim']))


    def test_sentence_rnn(self):
        with self.graph.as_default():
            X = self.graph.get_collection('input')[0]
            A = self.graph.get_collection('aspect')[0]
            sentence_rnn = self.graph.get_collection('sentence_rnn_output')[0]
            init = tf.global_variables_initializer()

        with tf.Session(graph=self.graph) as sess:
            sess.run(init)
            result_X,result_sentence_rnn=sess.run([X,sentence_rnn],feed_dict={X:self.data_sentence,A:self.data_aspect})
            self.assertEqual(result_X.shape,self.data_sentence.shape)
            self.assertEqual(len(result_sentence_rnn),self.rnn_config['sentence_num'])
            self.assertEqual(result_sentence_rnn[0].shape,(self.rnn_config['batch_size'],1,self.rnn_config['sentence_cell_size']))

    def test_sentence_concat_with_aspect(self):
        with self.graph.as_default():
            X = self.graph.get_collection('input')[0]
            A = self.graph.get_collection('aspect')[0]
            init = tf.global_variables_initializer()
            sent_concat_aspect= self.graph.get_collection('sentence_concat_aspect')[0]
        with tf.Session(graph=self.graph) as sess:
            sess.run(init)
            result_concat = sess.run(sent_concat_aspect,feed_dict={X:self.data_sentence,A:self.data_aspect})
            self.assertEqual(len(result_concat),self.rnn_config['sentence_num'])
            self.assertEqual(result_concat[0].shape,(self.rnn_config['batch_size'],1,self.rnn_config['sentence_cell_size']+self.rnn_config['aspect_rnn_cell_size']))

    def test_sentence_concat_to_doc(self):
        with self.graph.as_default():
            X = self.graph.get_collection('input')[0]
            A = self.graph.get_collection('aspect')[0]
            init = tf.global_variables_initializer()
            Ds = self.graph.get_collection('Ds')[0]
        with tf.Session(graph=self.graph) as sess:
            sess.run(init)
            result_concat = sess.run(Ds,feed_dict={X:self.data_sentence,A:self.data_aspect})
            self.assertEqual(result_concat.shape,(self.rnn_config['batch_size'],self.rnn_config['sentence_num'],self.rnn_config['sentence_cell_size']))

if __name__=="__main__":
    # test=ClassifierTest()
    # test.test_aspect_rnn()
    unittest.main()
