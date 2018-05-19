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

    def aspect_prob2true_label(self,aspect_prob,mask):
        """

        :param aspect_prob: shape=(batch size, attributes num+1)
        :param mask: (batch size*max review length, attributes num+1) 
        :return: (batch size*max review length, attributes num+1)
        """
        condition = tf.greater_equal(aspect_prob,self.nn_config['aspect_prob_threshold'])
        true_labels = tf.where(condition,tf.ones_like(aspect_prob,dtype='float32'),tf.zeros_like(aspect_prob,dtype='float32'))
        true_labels = tf.tile(tf.expand_dims(true_labels, axis=1),
                              multiples=[1, self.nn_config['max_review_length'], 1])
        true_labels = tf.reshape(true_labels,shape=(-1,self.nn_config['attributes_num']+1))
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
        
        :param aspect_prob: shape=(batch size, attributes num+1)
        :param graph: 
        :return: (batch size*max review length, attributes num+1)
        """
        # aspect_prob.shape = (batch size, max review length, attributes num)
        aspect_prob = tf.tile(tf.expand_dims(aspect_prob,axis=1),multiples=[1,self.nn_config['max_review_length'],1])
        return tf.reshape(aspect_prob,shape=(-1,self.nn_config['attributes_num']+1))

    def attribute_labels_input(self, graph):
        """

        :param graph: 
        :return: shape = (batch size, attributes number+1)
        """
        Y_att = tf.placeholder(shape=(None, self.nn_config['attributes_num']),
                               dtype='float32')
        graph.add_to_collection('Y_att', Y_att)
        # TODO: add non-attribute
        batch_size = tf.shape(Y_att)[0]
        non_attr = tf.ones((batch_size,1),dtype='float32')*self.nn_config['non_attr_prob']
        Y_att = tf.concat([Y_att,non_attr],axis=1)
        graph.add_to_collection('check', Y_att)
        return Y_att

    def mask_for_true_label(self, X):
        """
        
        :param X: 
        :return:(batch size,) 
        """
        X = tf.cast(X, dtype='float32')
        temp = tf.reduce_min(X, axis=1)
        ones = tf.ones_like(temp, dtype='float32') * self.nn_config['padding_word_index']
        is_one = tf.equal(temp, ones)
        mask = tf.where(is_one, tf.zeros_like(temp, dtype='float32'), tf.ones_like(temp, dtype='float32'))
        mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, self.nn_config['attributes_num']+1])
        return mask

    def relevance_prob_senti(self,senti_score,Y_senti,graph):
        """
        
        :param senti_score: (batch size*max_review_length, number of attributes+1,3)
        :param Y_senti: [batch_size*max review len, number of attributes + 1, 3]
        :return: (batch size*max review length, attributes num+1, 3)
        """
        #senti_socre.shape=(batch size, max_review_length, number of attributes+1, 3)
        senti_score = tf.reshape(senti_score, shape=(-1, self.nn_config['max_review_length'], self.nn_config['attributes_num']+1,3))
        # prob.shape = (batch size, attributes num+1, 3, max review length); p(x;a)
        rel_prob = tf.nn.softmax(tf.transpose(senti_score, perm=[0, 2,3,1]), axis=3)
        # prob.shape = (batch size,max review length, attributes num+1, 3)
        rel_prob = tf.transpose(rel_prob, perm=[0, 3, 1,2])
        # shape = (batch size*max review length, attributes num+1, 3)
        rel_prob=tf.reshape(rel_prob, shape=(-1, self.nn_config['attributes_num']+1,3))
        # shape = (batch size*max review length, attributes num+1, 3)
        rel_prob = tf.multiply(rel_prob,Y_senti)
        # shape = (batch size*max review length, attributes num+1)
        rel_prob = tf.reduce_sum(rel_prob,axis=2)
        return rel_prob

    def softmax_loss(self, labels, logits,senti_rel_prob,aspect_prob, graph):
        """

        :param labels: (batch size*max review len, number of attributes+1,3)
        :param logits: (batch_size*max review len, number of attributes + 1, 3)
        :param senti_rel_prob:(batch_size*max review len, number of attributes+1)
        :param aspect_prob: (batch_size*max review len, number of attributes+1)
        :return: 
        """

        loss = tf.reduce_mean(tf.add(
            tf.reduce_sum(tf.multiply(aspect_prob,tf.multiply(senti_rel_prob,tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, dim=-1))), axis=1),
            tf.reduce_sum(graph.get_collection('reg'))))
        graph.add_to_collection('senti_loss', loss)
        return loss

    def senti_label2true_label(self,senti_label,mask):
        """
        mask padded sentences
        :param senti_label: shape=(batch size*max review len, attributes num+1,3)
        :param mask: (batch size*max review length, attributes num+1) 
        :return: (batch size*max review length, attributes num+1,3)
        """
        # mask.shape=(batch size*max review len, attributes num+1,3)
        mask = tf.tile(tf.expand_dims(mask,axis=2),multiples=[1,1,3])
        senti_label = senti_label * mask
        tf.add_to_collection('true_senti_labels',senti_label)
        return senti_label

    def expand_senti_label(self, senti_label,graph):
        """

        :param aspect_prob: shape=(batch size, attributes num+1,3)
        :param graph: 
        :return: (batch size*max review length, attributes num+1,3)
        """
        # aspect_prob.shape = (batch size, max review length, attributes num)
        aspect_prob = tf.tile(tf.expand_dims(senti_label, axis=1),
                              multiples=[1, self.nn_config['max_review_length'], 1,1])
        return tf.reshape(aspect_prob, shape=(-1, self.nn_config['attributes_num'] + 1,3))







