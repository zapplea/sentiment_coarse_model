import unittest
import numpy as np
import tensorflow as tf

from multi_filter import MultiFilter

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
        self.graph = tf.Graph()
        self.mf = MultiFilter(self.nn_config)

    def test_filter_generator(self):
        X = np.random.randn(10,self.nn_config['words_num']).astype('float32')
        sess = tf.Session()
        for filter_size in self.nn_config['filter_size']:
            filter =self.mf.filter_generator(X,filter_size)
            print(sess.run(filter))
            print('===========')

if __name__=="__main__":
    unittest.main()