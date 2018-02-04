import unittest
from classifier import AttributeFunction
import tensorflow as tf
import numpy as np

class AFTest(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(AFTest, self).__init__(*args, **kwargs)
        init={}
        self.nn_config={'attributes_num':20,
                        'attribute_senti_prototype_num':None,
                        'normal_senti_prototype_num':None, # number of specific sentiment of each attribute
                        'sentiment_dim':None, # dim of a sentiment expression prototype.
                        'attribute_dim':None,
                        'attribute_mat_size':3,
                        'words_num':10,
                        'word_dim':30,
                        'attribute_loss_theta':1.0,
                        'sentiment_loss_theta':1.0,
                        'is_mat':True,
                        'epoch':None,
                        'rps_num':None,
                        'rp_dim':None,
                        'lr':0.003,
                        'batch_size':30,
                        'atr_threshold':0}
        self.graph=tf.Graph()
        self.af=AttributeFunction(self.nn_config)
        # with self.graph.as_default():
        #     self.init=tf.global_variables_initializer()
        self.sess=tf.Session(graph=self.graph)

        # data
        #self.x=np.random.uniform(size=(self.nn_config['words_num'],self.nn_config['word_dim'])).astype('float32')
        self.x=np.ones(shape=(self.nn_config['words_num'],self.nn_config['word_dim']),dtype='float32')
        # attributes matrix
        self.A_mat = np.ones(shape = (self.nn_config['attributes_num'],self.nn_config['attribute_mat_size'],self.nn_config['word_dim']))
        self.o_mat = np.ones(shape=(self.nn_config['attribute_mat_size'],self.nn_config['word_dim']),dtype='float32')
        self.atr_labels= np.random.randint(0,1,size=(self.nn_config['batch_size'],self.nn_config['attributes_num'])).astype(dtype='float32')

    def test_attribute_vec(self):
        with self.graph.as_default():
            A,o = self.af.attribute_vec(self.graph)
            init = tf.global_variables_initializer()
        with self.sess:
            self.sess.run(init)
            A = self.sess.run(A)
            o = self.sess.run(o)
            self.assertEqual(np.array(A).shape,(self.nn_config['attributes_num'],self.nn_config['word_dim']))
            self.assertEqual(o.shape, (self.nn_config['word_dim'],))

    def test_attribute_mat(self):
        with self.graph.as_default():
            A_mat,o_mat=self.af.attribute_mat(self.graph)
            init = tf.global_variables_initializer()
        with self.sess:
            self.sess.run(init)
            A_mat = self.sess.run(A_mat)
            o_mat = self.sess.run(o_mat)
            self.assertEqual(np.array(A_mat).shape,(self.nn_config['attributes_num'],self.nn_config['attribute_mat_size'],self.nn_config['word_dim']))
            self.assertEqual(o_mat.shape,(self.nn_config['attribute_mat_size'],self.nn_config['word_dim']))

    def test_shape_attribute_mat_attention(self):
        attention = []
        with self.graph.as_default():
            A,o=self.af.attribute_mat(self.graph)
            init=tf.global_variables_initializer()
            word_embed=self.x[0]
            for att_mat in A:
                attention.append(self.af.attribute_mat_attention(att_mat,word_embed,self.graph))
        with tf.Session(graph=self.graph) as sess:
            sess.run(init)
            result = sess.run(attention)
        self.assertIsInstance(result,list)
        self.assertEqual(np.array(result).shape,(self.nn_config['attributes_num'],
                                                    self.nn_config['attribute_mat_size'],
                                                    self.nn_config['word_dim']))
        ## attribute_mat_attention can be convertated to matrix version. A_mat as input

    def test_value_attribute_mat_attention(self):
        a_mat = self.A_mat[0]
        word_embed = self.x[0]
        true_value = np.ones(shape=(self.nn_config['attribute_mat_size'],self.nn_config['word_dim']),dtype='float32')*(1/self.nn_config['attribute_mat_size'])
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            attention = self.af.attribute_mat_attention(a_mat,word_embed,self.graph)
        with tf.Session(graph=self.graph) as sess:
            sess.run(init)
            result = sess.run(attention)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                self.assertAlmostEqual(result[i][j],true_value[i][j],places=5)

    def test_shape_attribute_mat2vec(self):
        word_embed=self.x[0]
        with self.graph.as_default():
            A,o = self.af.attribute_mat(self.graph)
            A_vec,o_vec = self.af.attribute_mat2vec(word_embed,A,o,self.graph)
            init = tf.global_variables_initializer()
        with tf.Session(graph=self.graph) as sess:
            sess.run(init)
            result_A = sess.run(A_vec)
            result_o = sess.run(o_vec)
        self.assertEqual(np.array(result_A).shape,(self.nn_config['attributes_num'],self.nn_config['word_dim']))
        self.assertEqual(result_o.shape,(self.nn_config['word_dim'],))

    def test_value_attribute_mat2vec(self):
        word_embed = self.x[0]
        true_A = np.ones(shape=(self.nn_config['attributes_num'],self.nn_config['word_dim']),dtype='float32')
        true_o = np.ones(shape=(self.nn_config['word_dim'],),dtype = 'float32')
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            A_vec, o_vec = self.af.attribute_mat2vec(word_embed,self.A_mat,self.o_mat,self.graph)

        with tf.Session(graph=self.graph) as sess:
            sess.run(init)
            result_A = sess.run(A_vec)
            result_o = sess.run(o_vec)
        for i in range(self.nn_config['attributes_num']):
            for j in range(self.nn_config['word_dim']):
                self.assertLess(true_A[i][j]-result_A[i][j],0.0001)
        for j in range(self.nn_config['word_dim']):
            self.assertLess(true_o[j]-result_o[j],0.0001)

    def test_shape_score(self):
        with self.graph.as_default():
            A,o = self.af.attribute_mat(self.graph)
            score = self.af.score(A,o,self.x,self.graph)
            init = tf.global_variables_initializer()
        with tf.Session(graph=self.graph) as sess:
            sess.run(init)
            result = sess.run(score)
        self.assertEqual(result.shape,(self.nn_config['attributes_num'],))

    def test_shape_loss(self):
        atr_label = self.atr_labels[0]
        with self.graph.as_default():
            A, o = self.af.attribute_mat(self.graph)
            score = self.af.score(A, o, self.x, self.graph)
            loss = self.af.loss(score,atr_label,self.graph)
            init = tf.global_variables_initializer()
        with tf.Session(graph = self.graph) as sess:
            sess.run(init)
            result = sess.run(loss)
        self.assertEqual(result.shape,(self.nn_config['attributes_num'],))

if __name__=="__main__":
    unittest.main()