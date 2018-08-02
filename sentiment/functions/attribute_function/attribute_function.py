import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.functions.initializer.initializer import Initializer

import tensorflow as tf
import numpy as np

class AttributeFunction:
    def __init__(self, nn_config):
        self.nn_config = nn_config
        self.initializer=Initializer.parameter_initializer

    def attribute_vec(self, graph):
        """

        :param graph: 
        :return: shape = (number of attributes+1, attributes dim)
        """
        # A is matrix of attribute vector
        A = tf.Variable(initial_value=self.initializer(shape=(self.nn_config['attributes_num'],
                                                               self.nn_config['attribute_dim']),
                                                        dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(A))
        graph.add_to_collection('A_vec', A)
        o = tf.Variable(initial_value=self.initializer(shape=(1, self.nn_config['attribute_dim']),
                                                        dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(o))
        graph.add_to_collection('o_vec', o)
        return A, o

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

    def score(self, A, X, mask, graph):
        """

        :param A: shape = (number of attributes, attribute dim) or
                  shape = (batch size, words number, attributes num, attribute dim)
        :param X: shape = (batch size, words number, lstm cell size)
        :param graph: 
        :return: (batch size, attributes num, words num)
        """
        # TODO: should eliminate the influence of #PAD# when calculate reduce max
        if not self.nn_config['is_mat']:
            X = tf.reshape(X, shape=(-1, self.nn_config['lstm_cell_size']))
            # score.shape = (attributes num,batch size*words num)
            score = tf.matmul(A, X, transpose_b=True)
            # score.shape = (attributes num, batch size, words num)
            score = tf.reshape(score, (self.nn_config['attributes_num'], -1, self.nn_config['words_num']))
            # score.shape = (batch size, attributes number, words num)
            score = tf.transpose(score, [1, 0, 2])
            # mask.shape = (batch size, attributes number, words num)
            mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, self.nn_config['attributes_num'],1])
            score = tf.add(score, mask)
            # this part is put on the next
            # score.shape = (batch size, attributes num)
            # score = tf.reduce_max(score, axis=2)
        else:
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

    def prediction(self, score, graph):
        condition = tf.greater(score, tf.ones_like(score, dtype='float32') * self.nn_config['atr_pred_threshold'])
        pred = tf.where(condition, tf.ones_like(score, dtype='float32'), tf.zeros_like(score, dtype='float32'))
        graph.add_to_collection('atr_pred', pred)
        return pred

    # ######################### #
    # loss function for sigmoid #
    # ######################### #
    def sigmoid_loss(self, score, Y_att, graph):
        """

        :param score: shape=(batch size, attributes num)
        :return: 
        """
        loss = tf.reduce_mean(tf.add(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_att, logits=score),
                                                   axis=1),
                                     tf.reduce_sum(graph.get_collection('reg'))))
        tf.add_to_collection('atr_loss',loss)
        return loss
    # ############################ #
    # max margin loss function for multi-label#
    # ############################ #
    def max_false_score(self, score, Y_att, graph):
        """

        :param score: shape = (batch size, attributes num)
        :param Y_att: shape = (batch size, attributes num)
        :param graph: shape = (batch size,)
        :return: 
        """
        condition = tf.equal(Y_att, tf.ones_like(Y_att, dtype='float32'))
        max_fscore = tf.reduce_max(
            tf.where(condition, tf.ones_like(score) * tf.constant(-np.inf, dtype='float32'), score), axis=1)
        max_fscore = tf.where(tf.is_inf(max_fscore), tf.zeros_like(max_fscore, dtype='float32'), max_fscore)
        graph.add_to_collection('max_false_score', max_fscore)
        return max_fscore

    def sum_true_score(self,score, Y_att, graph):
        """
        
        :param score: shape = (batch size, attributes num)
        :param Y_att: shape = (batch size, attributes num)
        :param graph: 
        :return: shape=(batch size,)
        """
        sum_true = tf.reduce_sum(tf.multiply(score,Y_att), axis=1)
        return sum_true

    def max_margin_loss(self,max_fscore,sum_true,graph):
        """
        
        :param max_fscore: shape=(batch size, )
        :param sum_true: shape=(batch size, )
        :return: 
        """
        theta = tf.constant(self.nn_config['attribute_loss_theta'], dtype='float32')
        loss = tf.add(tf.subtract(theta,sum_true),max_fscore)
        zero_loss = tf.zeros_like(loss, dtype='float32')
        loss= tf.reduce_max(tf.concat([tf.expand_dims(loss,axis=1),tf.expand_dims(zero_loss,axis=1)]),axis=1)
        loss = tf.reduce_mean(tf.add(loss,tf.reduce_sum(graph.get_collection('reg'))))
        return loss

    # max margin for single label
    # def max_margin_loss(self, score, max_fscore, Y_att, graph):
    #     """
    #
    #     :param score: shape = (batch size, attributes num)
    #     :param max_fscore: shape = (batch size, attributes num)
    #     :param Y_att: (batch size, attributes num)
    #     :param graph:
    #     :return: (batch size, attributes number)
    #     """
    #     # create a mask for non-attribute in which Sa is 0 and need to keep max false attribute
    #     Y_temp = tf.reduce_sum(Y_att, axis=1)
    #     condition = tf.equal(tf.reduce_sum(Y_att, axis=1),
    #                          tf.zeros_like(Y_temp, dtype='float32'))
    #     item1 = tf.tile(tf.expand_dims(
    #         tf.multiply(tf.ones_like(Y_temp, dtype='float32'), tf.divide(1, self.nn_config['attributes_num'])), axis=1),
    #         multiples=[1, self.nn_config['attributes_num']])
    #     nonatr_mask = tf.where(condition, item1, Y_att)
    #     #
    #     theta = tf.constant(self.nn_config['attribute_loss_theta'], dtype='float32')
    #     # loss.shape = (batch size, attributes num)
    #     loss = tf.multiply(tf.add(tf.subtract(theta, tf.multiply(Y_att, score)),max_fscore), nonatr_mask)
    #     zero_loss = tf.zeros_like(loss, dtype='float32')
    #
    #     loss = tf.expand_dims(loss, axis=2)
    #     zero_loss = tf.expand_dims(zero_loss, axis=2)
    #     # loss.shape = (batch size, attributes num)
    #     loss = tf.reduce_max(tf.concat([loss, zero_loss], axis=2), axis=2)
    #     # The following is obsolete design of loss function with regularization.
    #     # loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1)) + tf.multiply(1 / self.nn_config['batch_size'],
    #     #                                                                  tf.reduce_sum(graph.get_collection('reg')))
    #     loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1) + tf.reduce_sum(graph.get_collection('reg')))
    #     graph.add_to_collection('atr_loss', loss)
    #     tf.summary.scalar('loss', loss)
    #
    #     return loss

    # ################################################### #
    # The following is common part for sentiment function #
    # ################################################### #
    def sentences_input(self, graph):
        X = tf.placeholder(
            shape=(None, self.nn_config['words_num']),
            dtype='int32')
        graph.add_to_collection('X', X)
        return X

    def attribute_labels_input(self, graph):
        Y_att = tf.placeholder(shape=(None, self.nn_config['attributes_num']), dtype='float32')
        graph.add_to_collection('Y_att', Y_att)
        return Y_att

    def sequence_length(self, X, graph):
        """

        :param X: (batch size, max words num)
        :param graph: 
        :return: (batch size,)
        """
        paddings = tf.ones_like(X, dtype='int32') * self.nn_config['padding_word_index']
        condition = tf.equal(paddings, X)
        seq_len = tf.reduce_sum(tf.where(condition, tf.zeros_like(X, dtype='int32'), tf.ones_like(X, dtype='int32')),
                                axis=1, name='seq_len')
        return seq_len

    def mask_for_pad_in_score(self, X, graph):
        """
        This mask is used in score, to eliminate the influence of pad words when reduce_max. This this mask need to add to the score.
        Since 0*inf = nan
        :param X: the value is word id. shape=(batch size, max words num)
        :param graph: 
        :return: 
        """
        paddings = tf.ones_like(X, dtype='int32') * self.nn_config['padding_word_index']
        condition = tf.equal(paddings, X)
        mask = tf.where(condition, tf.ones_like(X, dtype='float32') * (-np.inf), tf.zeros_like(X, dtype='float32'))
        return mask

        # should use variable share

    def sentence_lstm(self, X, seq_len, graph):
        """
        return a lstm of a sentence
        :param X: shape = (batch size, words number, word dim)
        :param seq_len: shape = (batch size,) show the number of words in a batch
        :param graph: 
        :return: 
        """
        keep_prob = self.nn_config['keep_prob_lstm']
        # graph.add_to_collection('keep_prob_lstm',keep_prob)
        cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size']),input_keep_prob=keep_prob , output_keep_prob=keep_prob,state_keep_prob=keep_prob)
        # outputs.shape = (batch size, max_time, cell size)
        outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=X, time_major=False, sequence_length=seq_len, dtype='float32')
        graph.add_to_collection('sentence_lstm_outputs', outputs)
        return outputs

    def sentence_bilstm(self, X, seq_len, graph):
        """
        return a lstm of a sentence
        :param X: shape = (batch size, words number, word dim)
        :param seq_len: shape = (batch size,) show the number of words in a batch
        :param graph: 
        :return: 
        """
        keep_prob = tf.placeholder(dtype='float32')
        graph.add_to_collection('keep_prob_lstm', keep_prob)
        fw_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(int(self.nn_config['lstm_cell_size']/2)),input_keep_prob=keep_prob , output_keep_prob=keep_prob,state_keep_prob=keep_prob)
        bw_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(int(self.nn_config['lstm_cell_size']/2)),input_keep_prob=keep_prob , output_keep_prob=keep_prob,state_keep_prob=keep_prob)
        # outputs.shape = [(batch size, max time step, lstm cell size/2),(batch size, max time step, lstm cell size/2)]
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=X,sequence_length=seq_len,dtype='float32')
        # outputs.shape = (batch size, max time step, lstm cell size)
        outputs = tf.concat(outputs, axis=2, name='bilstm_outputs')
        graph.add_to_collection('sentence_bilstm_outputs', outputs)
        graph.add_to_collection('reg',
                                tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                    graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')))
        graph.add_to_collection('reg',
                                tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                    graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')))
        return outputs

    def optimizer(self, loss, graph):
        opt = tf.train.AdamOptimizer(self.nn_config['lr']).minimize(loss)
        graph.add_to_collection('opt', opt)
        return opt

    def is_word_padding_input(self, X, graph):
        """
        To make the sentence have the same length, we need to pad each sentence with '#PAD#'. To avoid updating of the vector,
        we need a mask to multiply the result of lookup table.
        :param graph: 
        :return: shape = (batch size, words number, word dim)
        """
        X = tf.cast(X, dtype='float32')
        ones = tf.ones_like(X, dtype='float32') * self.nn_config['padding_word_index']
        is_one = tf.equal(X, ones)
        mask = tf.where(is_one, tf.zeros_like(X, dtype='float32'), tf.ones_like(X, dtype='float32'))
        mask = tf.tile(tf.expand_dims(mask, axis=2), multiples=[1, 1, self.nn_config['word_dim']])
        return mask

    def lookup_table(self, X, mask, graph):
        """
        :param X: shape = (batch_size, words numbers)
        :param mask: used to prevent update of #PAD#
        :return: shape = (batch_size, words numbers, word dim)
        """
        table = tf.placeholder(shape=(self.nn_config['lookup_table_words_num'], self.nn_config['word_dim']),
                               dtype='float32')
        graph.add_to_collection('table', table)
        table = tf.Variable(table, name='table')
        embeddings = tf.nn.embedding_lookup(table, X, partition_strategy='mod', name='lookup_table')
        embeddings = tf.multiply(embeddings, mask)
        graph.add_to_collection('lookup_table', embeddings)
        return embeddings

    def mask_for_true_label(self, X):
        X = tf.cast(X, dtype='float32')
        temp = tf.reduce_min(X, axis=1)
        ones = tf.ones_like(temp, dtype='float32') * self.nn_config['padding_word_index']
        is_one = tf.equal(temp, ones)
        mask = tf.where(is_one, tf.zeros_like(temp, dtype='float32'), tf.ones_like(temp, dtype='float32'))
        mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, self.nn_config['attributes_num']])
        return mask