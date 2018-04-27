import tensorflow as tf
import numpy as np
import math

class MultiFilter:
    def __init__(self,nn_config):
        self.nn_config = nn_config

    def filter_generator(self,X,filter_size):
        """
        
        :param filter_size: 
        :return: (words num, words num) 
        """
        filter = tf.ones_like(X, dtype='int32')
        filter_org = filter
        r = tf.range(start=0, limit=self.nn_config['words_num'], delta=1, dtype='int32')
        filter = filter * r
        r = tf.range(start=0, limit=tf.shape(filter)[0], delta=1, dtype='int32')
        r2 = r
        r = tf.tile(tf.expand_dims(r, axis=1), multiples=[1, self.nn_config['words_num']]) * self.nn_config['words_num']

        filter = filter + r
        filter = tf.tile(tf.expand_dims(filter, axis=2), multiples=[1, 1, filter_size])

        half = int(math.ceil(filter_size/2))
        kernel=[]
        for i in range(half):
            if i==0:
                kernel.append(0)
            else:
                kernel.append(i)
                kernel.insert(0,-i)

        filter = filter + tf.constant(kernel, dtype='int32')

        r_lowerbound = tf.tile(tf.expand_dims(r2, axis=1), multiples=[1, self.nn_config['words_num']]) * self.nn_config['words_num']
        r_upperbound = tf.add(r_lowerbound, self.nn_config['words_num']-1)
        r_lowerbound = tf.tile(tf.expand_dims(r_lowerbound, axis=2), multiples=[1, 1, filter_size])
        r_upperbound = tf.tile(tf.expand_dims(r_upperbound, axis=2), multiples=[1, 1, filter_size])

        condition_lower = tf.greater_equal(filter, r_lowerbound)
        condition_upper = tf.less_equal(filter, r_upperbound)
        condition = tf.logical_and(condition_lower, condition_upper)

        filter = tf.where(condition, filter, tf.ones_like(filter, dtype='int32') * tf.size(filter_org))
        return filter

    def look_up(self,X,filter):
        pass


    def convolution(self,X,filter):
        """
        
        :param X: shape=(batch size, max sentence length, word dim)
        :param filter: shape = (max sentence length, max sentence length)
        :return: 
        """
        pass


    def score(self, A, X, mask, graph):
        """

        :param A: shape = (number of attributes, attribute dim) or
                  shape = (batch size, max words number, attributes num, attribute dim)
        :param X: shape = (batch size, max words number, word dim)
        :param mask: shape = (batch size, max words number)
        :param graph: 
        :return: (batch size, attributes num)
        """
        # finished TODO: should eliminate the influence of #PAD# when calculate reduce max
        if not self.nn_config['is_mat']:
            X = tf.reshape(X, shape=(-1, self.nn_config['word_dim']))
            # score.shape = (attributes num,batch size*words num)
            score = tf.matmul(A, X, transpose_b=True)
            # score.shape = (attributes num, batch size, words num)
            score = tf.reshape(score, (self.nn_config['attributes_num'], -1, self.nn_config['words_num']))
            # score.shape = (batch size, attributes number, words num)
            score = tf.transpose(score, [1, 0, 2])
            score = self.convolution(score)
            # mask.shape = (batch size, attributes number, words num)
            mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, self.nn_config['attributes_num'], 1])
            score = tf.add(score, mask)
            # score.shape = (batch size, attributes num)
            score = tf.reduce_max(score, axis=2)
        else:
            # X.shape = (batch size, words num, attributes num, attribute dim)
            X = tf.tile(tf.expand_dims(X, axis=2), multiples=[1, 1, self.nn_config['attributes_num'], 1])
            # score.shape = (batch size, words num, attributes num)
            score = tf.reduce_sum(tf.multiply(A, X), axis=3)
            # score.shape = (batch size, attributes num, words num)
            score = tf.transpose(score, [0, 2, 1])
            score = self.convolution(score)
            # mask.shape = (batch size, attributes number, words num)
            mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, self.nn_config['attributes_num'], 1])
            score = tf.add(score, mask)
        return score

