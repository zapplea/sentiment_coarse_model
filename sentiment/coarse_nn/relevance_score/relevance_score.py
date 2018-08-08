import tensorflow as tf
import numpy as np

# TODO: need to use threshold to predict if the Y is 1
# TODO: need to eliminate the influence of padded sentence
class RelScore:
    def __init__(self,nn_config):
        self.nn_config = nn_config

    def reviews_input(self,graph):
        X = tf.placeholder(
            shape=(20, self.nn_config['max_review_length'], self.nn_config['words_num']),
            dtype='int32')
        graph.add_to_collection('X', X)
        X = tf.reshape(X,shape=(-1,self.nn_config['words_num']))
        return X

    def aspect_prob2true_label(self,aspect_prob,mask):
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

    def complement1_aspect_prob(self,Y_att, aspect_prob):
        """
        1-p(a|D)
        :param Y_att: 
        :param aspect_prob: 
        :return: 
        """
        complementor = tf.subtract(1,Y_att)
        aspect_prob = tf.abs(tf.subtract(complementor,aspect_prob))
        return aspect_prob

    def complement2_aspect_prob(self,Y_att, aspect_prob):
        """
        p(a|D)=1
        :param Y_att: 
        :param aspect_prob: 
        :return: 
        """
        condition = tf.equal(tf.zeros_like(Y_att),Y_att)
        aspect_prob = tf.where(condition, tf.ones_like(aspect_prob),aspect_prob)
        return aspect_prob



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

    def sigmoid_loss(self, score, Y_att, atr_rel_prob,aspect_prob,graph):
        """

        :param score: shape=(batch size, attributes num)
        :return: 
        """
        loss = tf.reduce_mean(tf.add(tf.reduce_sum(tf.multiply(aspect_prob,
                                                               tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_att, logits=score),
                                                                           atr_rel_prob)),
                                                   axis=1),
                                     tf.reduce_sum(graph.get_collection('reg'))))
        tf.add_to_collection('atr_loss',loss)
        return loss

    def expand_aspect_prob(self,aspect_prob,graph):
        """
        
        :param aspect_prob: shape=(batch size, attributes num)
        :param graph: 
        :return: (batch size*max review length, attributes num)
        """
        # aspect_prob.shape = (batch size, max review length, attributes num)
        aspect_prob = tf.tile(tf.expand_dims(aspect_prob,axis=1),multiples=[1,self.nn_config['max_review_length'],1])
        return tf.reshape(aspect_prob,shape=(-1,self.nn_config['attributes_num']))

    def relevance_prob_senti(self,H):
        pass

    def coarse_senti_score(self,senti_prob,rel_prob,senti_score):
        pass