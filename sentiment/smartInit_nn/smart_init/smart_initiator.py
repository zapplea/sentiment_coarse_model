import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.functions.initializer.initializer import Initializer


import tensorflow as tf

class SmartInitiator:
    def __init__(self, nn_config):
        self.nn_config = nn_config
        self.initializer = Initializer.parameter_initializer

    def smart_initiater(self,graph):
        """
        :param attributes: ndarray, shape=(attribute numbers ,2, attribute dim)
        :return: 
        """
        name_list_input=tf.placeholder(shape=(None,self.nn_config['attributes_num']),dtype='float32')
        return name_list_input

    def name_list_score(self,name_list,graph):
        """
        
        :param initial: shape = (batch size, attributes num) 
        :param graph: 
        :return: 
        """
        W = tf.Variable(initial_value=self.initializer(shape=(self.nn_config['attributes_num'],),
                                                       dtype='float32'),
                        dtype='float32')
        graph.add_to_collection('reg',tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W))
        score = tf.multiply(W,name_list)
        return score