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

    def look_up(self,X,filter,filter_size):
        """
        
        :param X: shape=(batch size, max sentence length, word dim)
        :param filter: shape = (batch size, max sentence length, filter size)
        :return: (batch size, max sentence length, filter_size*word dim) 
        """
        # table.shape=(batch size * max sentence length, word dim)
        table = tf.reshape(X,shape=(-1,self.nn_config['word_dim']))
        # X.shape = (batch size, max sentence length, filter size, word dim)
        X = tf.nn.embedding_lookup(table, filter, partition_strategy='mod', name='lookup_table')
        # X.shape = (batch size, max sentence length, filter_size*word dim)
        X = tf.reshape(X,shape=(-1,self.nn_config['words_num'],filter_size*self.nn_config['word_dim']))
        return X

    def forward_layer(self,H,shape,name,graph):
        """
        
        :param H: (batch size, filter size*word dim)
        :param shape: 
        :param name: 
        :param graph: 
        :return: 
        """
        W = tf.get_variable(name=name+'_w', initializer=tf.random_uniform(shape=shape,dtype='float32'))
        graph.add_to_collection('reg',W)
        bias = tf.get_variable(name=name+'_b', initializer=tf.random_uniform(shape=shape[1],dtype='float32'))
        hidden_layer = tf.tanh(tf.add(tf.matmul(H,W),bias))
        return hidden_layer

    def convolution(self,X, filter_size, graph):
        """
        
        :param X: shape=(batch size, max sentence length, filter_size*word dim)
        :return: 
        """
        in_dim=filter_size*self.nn_config['word_dim']
        H = tf.reshape(X,shape=(-1,filter_size*self.nn_config['word_dim']))
        count = 0
        for layer_dim in self.nn_config['conv_layer_dim']:
            name='conv'+str(count)
            out_dim=layer_dim
            shape=(in_dim,out_dim)
            H = self.forward_layer(H,shape,name,graph)
            in_dim=out_dim
            count+=1
        #H.shape = (batch size, words num, out dim)
        H = tf.reshape(H,(-1,self.nn_config['words_num'],out_dim))
        return H
