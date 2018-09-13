import tensorflow as tf

class Initializer:
    @staticmethod
    def parameter_initializer(shape,dtype='float32'):
        stdv=1/tf.sqrt(tf.constant(shape[-1],dtype=dtype))
        init = tf.random_uniform(shape,minval=-stdv,maxval=stdv,dtype=dtype,seed=1)
        return init

class AttributeFunction:
    def __init__(self, nn_config):
        self.nn_config = nn_config
        self.initializer=Initializer.parameter_initializer

    def attribute_mat(self, graph):
        """

        :param graph: 
        :return: shape = (attributes number+1, attribute mat size, attribute dim)
        """
        A_mat = tf.Variable(initial_value=self.initializer(shape=(self.nn_config['attributes_num'],
                                                                    self.nn_config['attribute_mat_size'],
                                                                    self.nn_config['attribute_dim']),
                                                            dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(A_mat))
        graph.add_to_collection('A_mat', A_mat)
        o_mat = tf.Variable(initial_value=self.initializer(shape=(1,
                                                                   self.nn_config['attribute_mat_size'],
                                                                   self.nn_config['attribute_dim']),
                                                            dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(o_mat))
        graph.add_to_collection('o_mat', o_mat)
        return A_mat,o_mat

    def words_attribute_mat2vec(self, H, A_mat, graph):
        """
        convert attribtes matrix to attributes vector for each words in a sentence. A_mat include non-attribute mention matrix.
        :param H: shape = (batch size, number of words, word dim)
        :param A_mat: (number of atr, atr mat size, atr dim)
        :param graph: 
        :return: shape = (batch size, number of words, number of attributes, attribute dim(=lstm cell dim))
        """
        # H.shape = (batch size, words number, attribute number, word dim)
        H = tf.tile(tf.expand_dims(H, axis=2), multiples=[1, 1, self.nn_config['attributes_num'], 1])
        # H.shape = (batch size, words number, attribute number, attribute mat size, word dim)
        H = tf.tile(tf.expand_dims(H, axis=3), multiples=[1, 1, 1, self.nn_config['attribute_mat_size'], 1])
        # attention.shape = (batch size, words number, attribute number, attribute mat size)
        attention = tf.nn.softmax(tf.reduce_sum(tf.multiply(H, A_mat), axis=4))
        # attention.shape = (batch size, words number, attribute number, attribute mat size, attribute dim)
        attention = tf.tile(tf.expand_dims(attention, axis=4), multiples=[1, 1, 1, 1, self.nn_config['attribute_dim']])
        words_A = tf.reduce_sum(tf.multiply(attention, A_mat), axis=3)
        graph.add_to_collection('words_attributes', words_A)
        return words_A

    def words_nonattribute_mat2vec(self, H, o_mat, graph):
        """

        :param H: shape = (batch size, words number, word dim)
        :param o_mat: shape = (1,attribute mat size, attribute dim)
        :param graph: 
        :return: batch size, number of words, attributes num, attribute dim( =word dim)
        """
        # H.shape = (batch size, words number, 1, word dim)
        H = tf.expand_dims(H, axis=2)
        # H.shape = (batch size, words number, 1, attribute mat size, word dim)
        H = tf.tile(tf.expand_dims(H, axis=3), multiples=[1, 1, 1, self.nn_config['attribute_mat_size'], 1])
        # attention.shape = (batch size, words number, 1, attribute mat size)
        attention = tf.nn.softmax(tf.reduce_sum(tf.multiply(H, o_mat), axis=4))
        # attention.shape = (batch size, words number, 1, attribute mat size, attribute dim)
        attention = tf.tile(tf.expand_dims(attention, axis=4), multiples=[1, 1, 1, 1, self.nn_config['attribute_dim']])
        # words_A.shape = (batch size, number of words, 1, attribute dim( =word dim))
        words_o = tf.reduce_sum(tf.multiply(attention, o_mat), axis=3)
        # words_A.shape = (batch size, number of words, attributes number, attribute dim( =word dim))
        words_o = tf.tile(words_o, multiples=[1, 1, self.nn_config['attributes_num'], 1])
        graph.add_to_collection('words_nonattribute', words_o)
        return words_o

    def sentence_score(self, A, X, mask, graph):
        """

        :param A: shape = (number of attributes, attribute dim) or
                  shape = (batch size, words number, attributes num, attribute dim)
        :param X: shape = (batch size, words number, lstm cell size)
        :param graph: 
        :return: (batch size, attributes num, words num)
        """
        # TODO: should eliminate the influence of #PAD# when calculate reduce max
        # X.shape = (batch size, words num, attributes num, attribute dim)
        X = tf.tile(tf.expand_dims(X, axis=2), multiples=[1, 1, self.nn_config['attributes_num'], 1])
        # score.shape = (batch size, words num, attributes num)
        score = tf.reduce_sum(tf.multiply(A, X), axis=3)
        # score.shape = (batch size, attributes num, words num)
        score = tf.transpose(score, [0, 2, 1])
        # mask.shape = (batch size, attributes number, words num)
        mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, self.nn_config['attributes_num'],1])
        score = tf.add(score, mask)
        # score.shape = (batch size, attributes num)
        # score = tf.reduce_max(score, axis=2)
        return score

    def sentence_sigmoid(self,score,graph):
        """
        
        :param score: shape=(batch size*max review length, attributes num)
        :param graph: 
        :return: 
        """
        return tf.nn.sigmoid(score)

    def review_mask(self,X):
        """
        mask padded sentence in review_sigmoid()
        :param X: shape=(batch size, max review length, wrods num)
        :param graph: 
        :return:  shape = (batch size*max review length, coarse attributes num)
        """
        paddings = tf.ones_like(X, dtype='int32') * self.nn_config['padding_word_index']
        condition = tf.equal(paddings, X)
        condition = tf.reduce_all(condition, axis=2)
        #shape = (batch size, max review length)
        mask = tf.where(condition, tf.zeros_like(condition, dtype='float32'), tf.ones_like(condition, dtype='float32'))

        mask = tf.reshape(tf.tile(tf.expand_dims(mask,axis=2),multiples=[1,1,self.nn_config['coarse_attributes_num']]),
                          shape=(-1,self.nn_config['coarse_attributes_num']))
        return mask

    def review_score(self,sentence_prob,mask,review_len,graph):
        """
        
        :param sentence_prob: shape = (batch size * max review length, attributes num)
        :param mask: shape = (batch size*max review length, coarse attributes num)
        :param review_len: # shape = (batch size,)
        :param graph: 
        :return: (batch size, coarse_attributes_num)
        """
        W = tf.get_variable(name='W_trans',
                            initializer=self.initializer(shape=(self.nn_config['coarse_attributes_num'],
                                                                self.nn_config['attributes_num']),
                                                         dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W))
        # shape = (batch size * max review length, coarse_attributes_num, attributes num)
        sentence_prob = tf.tile(tf.expand_dims(sentence_prob,axis=1),multiples=[1,self.nn_config['coarse_attributes_num'],1])
        # shape = (batch size * max review length, coarse_attributes_num)
        sentence_prob = tf.reduce_sum(tf.multiply(W,sentence_prob),axis=2)
        # shape = (batch size * max review length, coarse_attributes_num)
        sentence_prob = tf.multiply(sentence_prob,mask)
        # shape = (batch size, max review length, coarse_attributes_num)
        sentence_prob = tf.reshape(sentence_prob,shape = (-1,self.nn_config['max_review_len'],self.nn_config['coarse_attributes_num']))
        # shape = (batch size, coarse attributes num)
        review_len = tf.tile(tf.expand_dims(review_len,axis=1),multiples=[1,self.nn_config['coarse_attributes_num']])
        # shape = (batch size, coarse_attributes_num)
        review_score = tf.truediv(tf.reduce_sum(sentence_prob,axis=1),review_len)

        return review_score

    def sigmoid_loss(self, score, Y_att, graph):
        """

        :param score: shape=(batch size*max review length, attributes num)
        :return: 
        """
        loss = tf.reduce_mean(tf.add(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_att, logits=score),
                                                   axis=1),
                                     tf.reduce_sum(graph.get_collection('reg'))))
        tf.add_to_collection('atr_loss',loss)
        return loss

    def prediction(self, score, graph):
        """
        :param score: shape = (batch size, coarse attributes num) 
        :param graph: 
        :return: 
        """
        #
        prob = tf.sigmoid(score)
        condition = tf.greater(prob,self.nn_config['review_atr_pred_threshold'])
        pred = tf.where(condition, tf.ones_like(score, dtype='float32'), tf.zeros_like(score, dtype='float32'))
        graph.add_to_collection('atr_pred', pred)
        return pred


class SentimentFunction:
    def __init__(self,nn_config):
        self.nn_config = nn_config