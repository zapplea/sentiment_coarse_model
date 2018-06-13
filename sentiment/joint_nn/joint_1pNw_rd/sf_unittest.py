import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
else:
    sys.path.append('/home/liu121/sentiment_coarse_model')


from sentiment.joint_nn.joint_1pNw_rd.classifier import Classifier

import unittest
import tensorflow as tf
import numpy as np
import math

class DataGenerator:
    def __init__(self,dg):
        self.dg=dg

class SFTest(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(SFTest, self).__init__(*args, **kwargs)
        seed = {'lstm_cell_size': 300,
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
                          'epoch': 10000,
                          'rps_num': 5,  # number of relative positions
                          'rp_dim': 300,  # dimension of relative position
                          'lr': 0.003,  # learing rate for attribute loss,
                          'senti_lr':0.003,
                          'joint_lr':0.003,
                          'batch_size': 30,
                          'lstm_cell_size': seed['lstm_cell_size'],
                          'atr_threshold': 0,  # attribute score threshold
                          'reg_rate': 0.03,
                          'atr_pred_threshold':0,
                          'lookup_table_words_num':2981402, # 2074276 for Chinese word embedding
                          'padding_word_index':1,
                          'rel_word_dim':300,
                          'rel_words_num':10
                          }
        self.graph=tf.Graph()
        self.dg = DataGenerator(self.nn_config)
        self.sess=tf.Session(graph=self.graph)
        self.cf = Classifier(self.nn_config,self.dg)

    def test_classifier(self):
        graph,saver = self.cf.classifier()


if __name__ == "__main__":
    unittest.main()