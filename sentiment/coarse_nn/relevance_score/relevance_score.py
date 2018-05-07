import tensorflow as tf
import numpy as np

# TODO: need to use threshold to predict if the Y is 1
# TODO: need to eliminate the influence of padded sentence
class RelScore:
    def __init__(self,nn_config):
        self.nn_config = nn_config

    def reviews_input(self,graph):
        X = tf.placeholder(
            shape=(None, self.nn_config['max_review_length'], self.nn_config['words_num']),
            dtype='int32')
        graph.add_to_collection('X', X)
        X = tf.reshape(X,shape=(-1,self.nn_config['words_num']))
        return X

    def aspect_prob2true_label(self,aspect_prob ,mask):
        """

        :param aspect_prob: shape=(batch size, attributes num)
        :param graph: 
        :return: (batch size*max review length, attributes num)
        """
        condition = tf.greater_equal(aspect_prob,self.nn_config['aspect_prob_threshold'])
        true_labels = tf.where(condition,tf.ones_like(aspect_prob,dtype='float32'),tf.zeros_like(aspect_prob,dtype='float32'))
        true_labels = tf.tile(tf.expand_dims(true_labels, axis=1),
                              multiples=[1, self.nn_config['max_review_length'], 1])
        true_labels = tf.reshape(true_labels,shape=(-1,self.nn_config['attributes_num']))
        true_labels = true_labels * mask
        tf.add_to_collection('true_labels',true_labels)
        return true_labels

    # def sentence_lstm(self, X, seq_len, graph):
    #     """
    #     return a lstm of a sentence
    #     :param X: shape = (batch size * max review length, words number, word dim)
    #     :param seq_len: shape = (batch,) show the number of words in a batch
    #     :param graph:
    #     :return: outputs.shape = (batch size * max review length, words number, word dim);
    #              last_hidden.shape = (batch size * max review length, word dim)
    #     """
    #     cell = tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size'])
    #     # outputs.shape = (batch size, max_time, cell size)
    #     outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=X, time_major=False, sequence_length=seq_len, dtype='float32')
    #     graph.add_to_collection('sentence_lstm_outputs', outputs)
    #     # get the last lifted embedding for each sentence
    #     index0 = tf.cast(tf.expand_dims(tf.range(start=0,
    #                                              limit=self.nn_config['max_review_length']*self.nn_config['batch_size']),
    #                                     axis=1),
    #                      dtype='int32')
    #     index1 = tf.cast(tf.expand_dims(tf.expand_dims(seq_len-1,axis=1)),dtype='int32')
    #     slice_index = tf.concat([index0,index1],axis=1)
    #     last_hidden = tf.gather(outputs,indices=slice_index)
    #     return outputs, last_hidden



    def relevance_prob_atr(self, atr_score, graph):
        """
        P(x|a)
        :param atr_score: (batch size*max review length, attributes num)
        :return: shape = (batch size*max review length, attributes num) , in dimension 2 values are the same
        """
        atr_score = tf.reshape(atr_score,shape=(-1, self.nn_config['max_review_length'], self.nn_config['attributes_num']))
        # prob.shape = (batch size, attributes num, max review length); p(x;a)
        rel_prob = tf.nn.softmax(tf.transpose(atr_score,perm=[0,2,1]),axis=2)
        # prob.shape = (batch size,max review length, attributes num)
        rel_prob = tf.transpose(rel_prob,perm=[0,2,1])
        return tf.reshape(rel_prob,shape=(-1,self.nn_config['attributes_num']))

    def expand_aspect_prob(self,aspect_prob,graph):
        """
        
        :param aspect_prob: shape=(batch size, attributes num)
        :param graph: 
        :return: (batch size*max review length, attributes num)
        """
        # aspect_prob.shape = (batch size, max review length, attributes num)
        aspect_prob = tf.tile(tf.expand_dims(aspect_prob,axis=1),multiples=[1,self.nn_config['max_review_length'],1])
        return tf.reshape(aspect_prob,shape=(-1,self.nn_config['attributes_num']))

    def coarse_atr_score(self,aspect_prob,rel_prob,atr_score):
        """
        
        :param aspect_prob: (batch size*max review length, attributes num)
        :param rel_prob: (batch size*max review length, attributes num)
        :param atr_score: (batch size*max review length, attributes num)
        :return: (batch size*max review length, attributes num)
        """
        condition = tf.equal(atr_score,tf.convert_to_tensor(-np.inf))
        score = tf.multiply(rel_prob,tf.multiply(aspect_prob,atr_score))
        score = tf.where(condition,tf.ones_like(score)*(-np.inf),score)
        tf.add_to_collection('coarse_atr_score',score)
        return score

    def relevance_prob_senti(self,H):
        pass

    def coarse_senti_score(self,senti_prob,rel_prob,senti_score):
        pass