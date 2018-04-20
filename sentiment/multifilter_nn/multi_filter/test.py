import unittest
import numpy as np
import tensorflow as tf

from multi_filter import MultiFilter

class Test(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        print('in test')
        super(Test,self).__init__(*args,**kwargs)
        self.nn_config = {
            'words_num':10,
            'filter_size':[1,3,5]
        }
        self.graph = tf.Graph()


    def test_convolution(self):
        score = np.ones(shape=(10,6,10),dtype='float32')

        with self.graph.as_default():
            print('before convolution')
            mf = MultiFilter(self.nn_config)
            score_ls = mf.convolution(score)
            sess=tf.Session()
            for score,filter in score_ls:
                score_result,filter_result = sess.run([score,filter])
                print(score_result)
                print(filter_result)
                print('==================')

if __name__=="__main__":
    unittest.main()