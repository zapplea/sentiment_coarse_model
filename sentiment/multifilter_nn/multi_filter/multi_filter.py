import tensorflow as tf
import numpy as np
import math

class MultiFilter:
    def __init__(self,nn_config):
        self.nn_config = nn_config

    def filter_generator(self,filter_size):
        """
        
        :param filter_size: 
        :return: (words num, words num) 
        """
        filter = np.zeros(shape=(self.nn_config['words_num'],self.nn_config['words_num']),dtype='float32')
        for i in range(self.nn_config['words_num']):
            half = int(math.ceil(filter_size/2))
            for j in range(half):
                if i+j<self.nn_config['words_num']:
                    filter[i][i+j]=1
                if i-j>=0:
                    filter[i][i-j]=1
        return filter


    def convolution(self,score):
        """
        
        :param score: (batch size, attributes number, words num); Tensor
        :param filter_size: ndarray
        :return: 
        """
        score_ls = []
        for filter_size in self.nn_config['filter_size']:
            filter = self.filter_generator(filter_size=filter_size)
            filter = tf.constant(filter)
            # score.shape = (batch size, attributes number, words num, words num)
            new_score = tf.tile(tf.expand_dims(score,axis=2),multiples=[1,1,self.nn_config['words_num'],1])
            new_score = tf.multiply(new_score,filter)
            # score.shape = (batch size, attributes number, words num)
            new_score = tf.reduce_sum(new_score,axis=3)
            score_ls.append(new_score)
        # score.shape=(batch size, attributes number, words num)
        new_score = tf.add_n(score_ls)
        return new_score

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
            # score.shape = (batch size, attributes num)
            score = tf.reduce_max(score, axis=2)
        graph.add_to_collection('score', score)
        return score

    def score_1pNw(self, A, X,mask, graph):
        """

        :param A: shape = (number of attributes, attribute dim) or
                  shape = (batch size, words number, attributes num, attribute dim)
        :param X: shape = (batch size, words number, lstm cell size)
        :param graph: 
        :return: (batch size, attributes num)
        """
        if not self.nn_config['is_mat']:
            X = tf.reshape(X, shape=(-1, self.nn_config['lstm_cell_size']))
            # score.shape = (attributes num,batch size*words num)
            score = tf.matmul(A, X, transpose_b=True)
            # score.shape = (attributes num, batch size, words num)
            score = tf.reshape(score, (self.nn_config['attributes_num'], -1, self.nn_config['words_num']))
            # score.shape = (batch size, attributes number, words num)
            score = tf.transpose(score, [1, 0, 2])
            score = self.convolution(score)
            # mask.shape = (batch size, attributes number, words num)
            # use mask to eliminate the influence of
            mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, self.nn_config['attributes_num'],1])
            score = tf.add(score, mask)
        else:
            # X.shape = (batch size, words num, attributes num, attribute dim)
            X = tf.tile(tf.expand_dims(X, axis=2), multiples=[1, 1, self.nn_config['attributes_num'], 1])
            # score.shape = (batch size, words num, attributes num)
            score = tf.reduce_sum(tf.multiply(A, X), axis=3)
            # score.shape = (batch size, attributes num, words num)
            score = tf.transpose(score, [0, 2, 1])
            score = self.convolution(score)
            # mask.shape = (batch size, attributes number, words num)
            mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, self.nn_config['attributes_num'],1])
            score = tf.add(score, mask)
        return score