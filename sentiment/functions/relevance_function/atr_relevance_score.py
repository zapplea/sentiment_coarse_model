import tensorflow as tf


# TODO: need to use threshold to predict if the Y is 1
# TODO: need to eliminate the influence of padded sentence
class RelScore:
    def __init__(self, nn_config):
        self.nn_config = nn_config

    def reviews_input(self, graph):
        X = tf.placeholder(
            shape=(None, self.nn_config['max_review_length'], self.nn_config['words_num']),
            dtype='int32')
        graph.add_to_collection('X', X)
        X = tf.reshape(X, shape=(-1, self.nn_config['words_num']))
        return X

    def aspect_prob2true_label(self, aspect_prob, mask):
        """

        :param aspect_prob: shape=(batch size, attributes num)
        :param graph: 
        :return: (batch size*max review length, attributes num)
        """
        condition = tf.greater_equal(aspect_prob, self.nn_config['aspect_prob_threshold'])
        true_labels = tf.where(condition, tf.ones_like(aspect_prob, dtype='float32'),
                               tf.zeros_like(aspect_prob, dtype='float32'))
        true_labels = tf.tile(tf.expand_dims(true_labels, axis=1),
                              multiples=[1, self.nn_config['max_review_length'], 1])
        true_labels = tf.reshape(true_labels, shape=(-1, self.nn_config['attributes_num']))
        true_labels = true_labels * mask
        tf.add_to_collection('true_labels', true_labels)
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

    def softmax(self, score, mask):
        print('relevance score softmax')
        """

        :param score: (batch size, attributes num, max review length)
        :return: 
        """
        # shape = (batch size, attributes num, max review length)
        exp_score = tf.multiply(tf.exp(score),mask)
        # (batch size, attributes num)
        sum_score = tf.reduce_sum(exp_score,axis=2)
        # (batch size, attributes num, max review length)
        sum_score = tf.tile(tf.expand_dims(sum_score,axis=2),multiples=[1,1,self.nn_config['max_review_length']])
        rel_prob = tf.truediv(exp_score,sum_score)
        return rel_prob

    def relevance_prob_atr(self, atr_score, mask, graph):
        """
        P(x|a)
        :param atr_score: (batch size*max review length, attributes num)
        :param mask: (batch size*max review length, attributes num)
        :return: shape = (batch size*max review length, attributes num) , in dimension 2 values are the same
        """

        atr_score = tf.reshape(atr_score,shape=(-1, self.nn_config['max_review_length'], self.nn_config['attributes_num']))

        mask = tf.reshape(mask,shape=(-1, self.nn_config['max_review_length'], self.nn_config['attributes_num']))

        # prob.shape = (batch size, attributes num, max review length); p(x;a)
        # TODO: problem: the padded sentences is not masked.
        rel_prob = self.softmax(tf.transpose(atr_score, perm=[0, 2, 1]),tf.transpose(mask,perm=[0,2,1]))
        # prob.shape = (batch size,max review length, attributes num)
        rel_prob = tf.transpose(rel_prob, perm=[0, 2, 1])
        return tf.reshape(rel_prob, shape=(-1, self.nn_config['attributes_num']))

    def expand_aspect_prob(self, aspect_prob, graph):
        """

        :param aspect_prob: shape=(batch size, attributes num)
        :param graph: 
        :return: (batch size*max review length, attributes num)
        """
        # aspect_prob.shape = (batch size, max review length, attributes num)
        aspect_prob = tf.tile(tf.expand_dims(aspect_prob, axis=1),
                              multiples=[1, self.nn_config['max_review_length'], 1])
        return tf.reshape(aspect_prob, shape=(-1, self.nn_config['attributes_num']))

    # TODO: mask loss of padded sentences
    def sigmoid_loss(self, score, Y_att, atr_rel_prob, aspect_prob, mask, graph):
        """
        :param score: shape=(batch size*max review length, attributes num)
        :return: 
        """
        # mask loss of padded sentences

        loss = tf.reduce_mean(tf.add(tf.reduce_sum(tf.multiply(aspect_prob,
                                                               tf.multiply(
                                                                   tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_att,
                                                                                                                       logits=score),
                                                                               mask),
                                                                   atr_rel_prob)),
                                                   axis=1),
                                     tf.reduce_sum(graph.get_collection('reg'))))
        tf.add_to_collection('atr_loss', loss)
        return loss

    def sigmoid_loss_v2(self, score, Y_att, atr_rel_prob, aspect_prob, mask, graph):
        """
        In this loss function, we add p(x|a)logp(x|a)
        :param score: shape=(batch size, attributes num)
        :return: 
        """
        # mask loss of padded sentences
        rel_prob_reg = tf.multiply(tf.log(atr_rel_prob),atr_rel_prob)
        loss = tf.reduce_mean(tf.add(tf.reduce_sum(tf.add(tf.multiply(tf.multiply(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_att,
                                                                                                                                      logits=score),
                                                                                              mask),
                                                                                  atr_rel_prob),
                                                                      aspect_prob),
                                                          -rel_prob_reg),
                                                   axis=1),
                                     tf.reduce_sum(graph.get_collection('reg'))))
        tf.add_to_collection('atr_loss', loss)
        return loss

    def kl_loss(self, score, atr_rel_prob, aspect_prob, graph):
        """
        This is KL divergence, used to measure the difference between 
        :param score: (batch size * max reviews length, attributes num)
        :param aspect_prob: (batch size * max reviews length, attributes num)
        :param graph: 
        :return: 
        """
        p_distribution = aspect_prob
        q_distribution = tf.nn.sigmoid(score)

        kld = tf.reduce_sum(
            tf.multiply(tf.multiply(q_distribution, tf.log(tf.truediv(q_distribution, p_distribution))), atr_rel_prob),
            axis=1)
        # TODO: need to refine the size of atr_rel_prob.
        loss = tf.reduce_mean(tf.add(kld
                                     , tf.reduce_sum(graph.get_collection('reg'))))
        tf.add_to_collection('atr_loss', loss)
        return loss