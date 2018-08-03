import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
elif getpass.getuser() == "lizhou":
    sys.path.append('/media/data2tb4/yibing2/sentiment_coarse_model/')
from sentiment.functions.initializer.initializer import Initializer
from sentiment.coarse_nn.relevance_score.relevance_score import RelScore
from sentiment.functions.ilp.ilp import AttributeIlp

import tensorflow as tf
import numpy as np
import copy

class Transfer:
    def __init__(self, coarse_nn_config, coarse_data_generator):
        self.coarse_nn_config = coarse_nn_config
        self.coarse_data_generator = coarse_data_generator

    # def softmax(self, score):
    #     """
    #
    #     :param score: (sentences number, attributes num)
    #     :return:
    #     """
    #     avg_score = tf.reduce_mean(score, axis=0)
    #     index = tf.argmax(tf.nn.softmax(avg_score, axis=0))
    #     return index

    def media_model(self):
        af = AttributeFunction(self.coarse_nn_config)
        graph = tf.Graph()
        with graph.as_default():
            relscore = RelScore(self.coarse_nn_config)
            # X_ids.shape = (batch size * max review length, words num)
            X_ids = relscore.reviews_input(graph=graph)
            words_pad_M = af.is_word_padding_input(X_ids, graph)
            X = af.lookup_table(X_ids, words_pad_M, graph)
            # lstm
            with tf.variable_scope('sentence_bilstm'):
                seq_len = af.sequence_length(X_ids, graph)
                # H.shape = (batch size, max_time, cell size)
                H = af.sentence_bilstm(X, seq_len, graph=graph)

            if not self.coarse_nn_config['is_mat']:
                A, o = af.attribute_vec(graph)
                A = A - o
            else:
                A, o = af.attribute_mat(graph)
                # A.shape = (batch size, words num, attributes number, attribute dim)
                A_lstm = af.words_attribute_mat2vec(H, A, graph)
                o_lstm = af.words_nonattribute_mat2vec(H, o, graph)
                A_lstm = A_lstm - o_lstm
                A_e = af.words_attribute_mat2vec(X, A, graph)
                o_e = af.words_nonattribute_mat2vec(X, o, graph)
                A_e = A_e - o_e
            if not self.coarse_nn_config['is_mat']:
                mask = af.mask_for_pad_in_score(X_ids, graph)
                score_lstm = af.score(A, H, mask, graph)
                # score.shape = (batch size, attributes num, words num)
                score_e = af.score(A, X, mask, graph)
            else:
                mask = af.mask_for_pad_in_score(X_ids, graph)
                score_lstm = af.score(A_lstm, H, mask, graph)
                # score.shape = (batch size, attributes num, words num)
                score_e = af.score(A_e, X, mask, graph)
            # score.shape = (batch size, attributes num, words num)
            score = tf.add(score_lstm, score_e)
            tf.add_to_collection('score_pre', score)
            # score.shape = (batch size, attributes num)
            score = tf.reduce_max(score, axis=2)
            tf.add_to_collection('score', score)
            # eliminate the influce of -inf when calculate relevance probability with softmax
            condition = tf.is_inf(score)
            score = tf.where(condition, tf.zeros_like(score), score)

        return graph


    def transfer(self, coarse_model, fine_dg):
        # ##################### #
        #      coarse model     #
        # ##################### #
        graph, saver = coarse_model.classifier()
        with graph.as_default():
            bilstm_fw_kernel = graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')
            bilstm_fw_bias = graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/bias:0')
            bilstm_bw_kernel =graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')
            bilstm_bw_bias = graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/bias:0')
            table = graph.get_collection('table')[0]
            if self.coarse_nn_config['is_mat']:
                A = graph.get_collection('A_mat')[0]
                O = graph.get_collection('o_mat')[0]
            else:
                A = graph.get_collection('A_vec')[0]
                O = graph.get_collection('o_vec')[0]
        with graph.device('/gpu:1'):
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            table_data = fine_dg.table
            with tf.Session(graph=graph,config=config) as sess:
                model_file = tf.train.latest_checkpoint(self.coarse_nn_config['sr_path'])
                saver.restore(sess, model_file)
                A_data, O_data, bilstm_fw_kernel_data, bilstm_fw_bias_data, bilstm_bw_kernel_data, bilstm_bw_bias_data =\
                    sess.run([A,O,bilstm_fw_kernel,bilstm_fw_bias,bilstm_bw_kernel,bilstm_bw_bias],feed_dict={table:table_data})
            # A_data.shape=(attributes num, mat size, attribute dim)
            A_data= np.reshape(A_data,newshape=(1,self.coarse_nn_config['attributes_num']*self.coarse_nn_config['attribute_mat_size'],self.coarse_nn_config['attribute_dim']))

        # ##################### #
        #      media model      #
        # ##################### #
        graph = self.media_model()
        with graph.as_default():
            # score_pre.shape = (batch size, 1, words num)
            score_pre = graph.get_collection('score_pre')[0]
            # X.shape = (batch size, word numbers)
            X = graph.get_collection('X')[0]
            # dropout
            keep_prob_lstm = graph.get_collection('keep_prob_lstm')[0]
            # attention.shape = (batch size, words number, 1, aspects num*aspect mat size)
            attention = graph.get_collection('attention')[0]

            # lstm
            for v in tf.all_variables():
                if v.name.startswith('sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0'):
                    bilstm_fw_kernel = v
                elif v.name.startswith('sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/bias:0'):
                    bilstm_fw_bias = v
                elif v.name.startswith('sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0'):
                    bilstm_bw_kernel = v
                elif v.name.startswith('sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/bias:0'):
                    bilstm_bw_bias = v
            table = graph.get_collection('table')[0]
            if self.coarse_nn_config['is_mat']:
                A = graph.get_collection('A_mat')[0]
                O = graph.get_collection('o_mat')[0]
            else:
                A = graph.get_collection('A_vec')[0]
                O = graph.get_collection('o_vec')[0]
            init = tf.global_variables_initializer()
        with graph.device('/gpu:1'):
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(graph=graph,config=config) as sess:
                # table.load(table_data,sess)
                sess.run(init,feed_dict={table:table_data})
                A.load(A_data,sess)
                O.load(O_data,sess)
                bilstm_fw_kernel.load(bilstm_fw_kernel_data,sess)
                bilstm_fw_bias.load(bilstm_fw_bias_data,sess)
                bilstm_bw_kernel.load(bilstm_bw_kernel_data,sess)
                bilstm_bw_bias.load(bilstm_bw_bias_data,sess)
                X_data_list = self.coarse_data_generator.fine_sentences(fine_dg.train_attribute_ground_truth,fine_dg.train_sentence_ground_truth)
                label_id = 0
                ilp_data={}
                for X_data in X_data_list:
                    # in source model, things will be different
                    # score_pre.shape = (batch size, 1, words num)
                    # attention.shape = (batch size, words number, 1, aspects num*aspect mat size)
                    score_pre_data,attention_data = sess.run([score_pre,attention],feed_dict={X:X_data,keep_prob_lstm:1.0})
                    ilp_data[label_id]={'score_pre':score_pre_data,'attention':attention_data}
                    label_id+=1

        ilp = AttributeIlp(ilp_data,self.coarse_nn_config['attributes_num'],self.coarse_nn_config['attribute_mat_size'])
        index_collection = ilp.attributes_vec_index()
        A_data = ilp.attributes_matrix(index_collection,A_data[0])


        # index = self.softmax(score)
        # # X_data.shape = (fine grained attributes number, number of sentences,1,words num)
        # X_data_list = self.coarse_data_generator.fine_sentences(fine_dg.train_attribute_ground_truth,fine_dg.train_sentence_ground_truth)
        # config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allow_growth = True
        # index_list = []
        # initializer_A_data = []
        # with tf.Session(graph=graph, config=config) as sess:
        #     model_file = tf.train.latest_checkpoint(self.coarse_nn_config['sr_path'])
        #     saver.restore(sess, model_file)
        #     for X_data in X_data_list:
        #         index_data,score_data = sess.run([index,score], feed_dict={X: X_data,keep_prob_lstm:1.0})
        #         print(index_data, np.mean(score_data, axis=0))
        #         index_list.append(index_data)
        #
        #     A_data, initializer_O_data = sess.run([A, O])
        #     for index_data in index_list:
        #         initializer_A_data.append(A_data[index_data.astype('int32')])
        #     bilstm_fw_kernel_data,bilstm_fw_bias_data=sess.run([bilstm_fw_kernel,bilstm_fw_bias])
        #     bilstm_bw_kernel_data, bilstm_bw_bias_data = sess.run([bilstm_bw_kernel, bilstm_bw_bias])
        init_data = {'init_A': A_data, 'init_O': O_data,
                     'init_bilstm_fw_kernel': bilstm_fw_kernel_data, 'init_bilstm_fw_bias': bilstm_fw_bias_data,
                     'init_bilstm_bw_kernel':bilstm_bw_kernel_data,'init_bilstm_bw_bias':bilstm_bw_bias_data, 'init_table':table_data,
                     'coarse_A': A}
        return init_data


class AttributeFunction:
    def __init__(self, nn_config):
        self.nn_config = nn_config
        self.initializer = Initializer.parameter_initializer

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
        A_mat = tf.Variable(initial_value=self.initializer(shape=(1,
                                                                  self.nn_config['attributes_num']*self.nn_config['attribute_mat_size'],
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
        return A_mat, o_mat

    def words_attribute_mat2vec(self, H, A_mat, graph):
        """
        convert attribtes matrix to attributes vector for each words in a sentence. A_mat include non-attribute mention matrix.
        :param H: shape = (batch size, number of words, word dim)
        :param A_mat: (number of atr, atr mat size, atr dim)
        :param graph: 
        :return: shape = (batch size, number of words, number of attributes, attribute dim(=lstm cell dim))
        """
        # H.shape = (batch size, words number, 1, word dim)
        H = tf.expand_dims(H, axis=2)
        # H.shape = (batch size, words number, 1, attributes number*attribute mat size, word dim)
        H = tf.tile(tf.expand_dims(H, axis=3), multiples=[1, 1, 1, self.nn_config['attributes_num']*self.nn_config['attribute_mat_size'], 1])
        # attention.shape = (batch size, words number,1, attribute number*attribute mat size)
        attention = tf.nn.softmax(tf.reduce_sum(tf.multiply(H, A_mat), axis=4))
        graph.add_to_collection('attention',attention)
        # attention.shape = (batch size, words number,1, attribute number*attribute mat size, attribute dim)
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
            mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, self.nn_config['attributes_num'], 1])
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
            mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, self.nn_config['attributes_num'], 1])
            score = tf.add(score, mask)
            # score.shape = (batch size, attributes num)
            # score = tf.reduce_max(score, axis=2)
        return score

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
        keep_prob = tf.placeholder(tf.float32)
        tf.add_to_collection('keep_prob_lstm', keep_prob)
        cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size']),
                                             input_keep_prob=keep_prob, output_keep_prob=keep_prob,
                                             state_keep_prob=keep_prob)
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
        keep_prob = tf.placeholder(tf.float32)
        tf.add_to_collection('keep_prob_lstm', keep_prob)
        fw_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(int(self.nn_config['lstm_cell_size'] / 2)),
                                                input_keep_prob=keep_prob, output_keep_prob=keep_prob,
                                                state_keep_prob=keep_prob)
        bw_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(int(self.nn_config['lstm_cell_size'] / 2)),
                                                input_keep_prob=keep_prob, output_keep_prob=keep_prob,
                                                state_keep_prob=keep_prob)
        # outputs.shape = [(batch size, max time step, lstm cell size/2),(batch size, max time step, lstm cell size/2)]
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=X,
                                                     sequence_length=seq_len, dtype='float32')
        # outputs.shape = (batch size, max time step, lstm cell size)
        outputs = tf.concat(outputs, axis=2, name='bilstm_outputs')
        graph.add_to_collection('sentence_bilstm_outputs', outputs)
        graph.add_to_collection('reg',
                                tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                    graph.get_tensor_by_name(
                                        'sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')))
        graph.add_to_collection('reg',
                                tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                    graph.get_tensor_by_name(
                                        'sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')))
        return outputs

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