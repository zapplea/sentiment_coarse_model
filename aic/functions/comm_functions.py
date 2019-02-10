"""include functions can be used both in sentiment recognization and attribute extraction"""
import tensorflow as tf
import numpy as np

class Initializer:
    @staticmethod
    def parameter_initializer(shape,dtype='float32'):
        stdv=1/tf.sqrt(tf.constant(shape[-1],dtype=dtype))
        init = tf.random_uniform(shape,minval=-stdv,maxval=stdv,dtype=dtype,seed=1)
        return init

class FineCommFunction:
    def __init__(self,nn_config):
        self.nn_config = nn_config

    def sentences_input(self, graph):
        X = tf.placeholder(
            shape=(None, self.nn_config['words_num']),
            dtype='int32')
        graph.add_to_collection('X',X)
        return X

    def attribute_labels_input(self, graph):
        Y_att = tf.placeholder(shape=(None, self.nn_config['attributes_num']), dtype='float32')
        graph.add_to_collection('Y_att',Y_att)
        return Y_att

    def sentiment_labels_input(self,graph):
        """
        :param graph: 
        :return: shape=[batch_size, number of attributes+1, 3], thus ys=[...,sentence[...,attj_senti[0,1,0],...],...]
        """
        Y_senti = tf.placeholder(shape=(None, self.nn_config['attributes_num']+1, self.nn_config['sentiment_num']),
                                 dtype='float32')
        # TODO: add non-attribute
        graph.add_to_collection('Y_senti', Y_senti)
        return Y_senti

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
        # TODO: add regularizer
        keep_prob_lstm = tf.placeholder(dtype='float32')
        graph.add_to_collection('keep_prob_lstm',keep_prob_lstm)
        # graph.add_to_collection('keep_prob_lstm',keep_prob)
        cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size']),input_keep_prob=self.keep_prob_lstm , output_keep_prob=self.keep_prob_lstm,state_keep_prob=self.keep_prob_lstm)
        # outputs.shape = (batch size, max_time, cell size)
        outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=X, time_major=False, sequence_length=seq_len, dtype='float32')
        return outputs

    def sentence_bilstm(self,name, X, seq_len, reg, graph,scope_name=''):
        """
        return a lstm of a sentence
        :param X: shape = (batch size, words number, word dim)
        :param seq_len: shape = (batch size,) show the number of words in a batch
        :param graph: 
        :return: 
        """
        keep_prob_bilstm = tf.placeholder(dtype='float32')
        graph.add_to_collection('keep_prob_bilstm', keep_prob_bilstm)

        fw_cell = tf.nn.rnn_cell.LSTMCell(int(self.nn_config['lstm_cell_size'] / 2), )
        bw_cell = tf.nn.rnn_cell.LSTMCell(int(self.nn_config['lstm_cell_size'] / 2), )

        # outputs.shape = [(batch size, max time step, lstm cell size/2),(batch size, max time step, lstm cell size/2)]
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=X,sequence_length=seq_len,dtype='float32')
        # outputs.shape = (batch size, max time step, lstm cell size)
        outputs = tf.concat(outputs, axis=2, name='bilstm_outputs')
        reg[name].append(tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                    graph.get_tensor_by_name(scope_name+'/bidirectional_rnn/fw/lstm_cell/kernel:0')))
        reg[name].append(tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                    graph.get_tensor_by_name(scope_name+'/bidirectional_rnn/bw/lstm_cell/kernel:0')))
        return outputs

    # def optimizer(self, loss, graph):
    #     opt = tf.train.AdamOptimizer(self.nn_config['lr']).minimize(loss)
    #     graph.add_to_collection('opt', opt)
    #     return opt

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

    def lookup_table(self, X, mask, table, graph):
        """
        :param X: shape = (batch_size, words numbers)
        :param mask: used to prevent update of #PAD#
        :return: shape = (batch_size, words numbers, word dim)
        """
        embeddings = tf.nn.embedding_lookup(table, X, partition_strategy='mod', name='lookup_table')
        embeddings = tf.multiply(embeddings, mask)
        return embeddings

class CoarseCommFunction:
    def __init__(self,nn_config):
        self.nn_config = nn_config
        self.initializer = Initializer.parameter_initializer

    def sentences_input(self,graph):
        X = tf.placeholder(shape=(None,self.nn_config['max_review_len'],self.nn_config['words_num']),dtype='int32')
        # X = tf.reshape(X,shape=(-1,self.nn_config['words_num']))
        graph.add_to_collection('X',X)
        return X

    def attribute_labels_input(self,graph):
        Y_ = tf.placeholder(shape=(None, self.nn_config['attributes_num']), dtype='float32')
        graph.add_to_collection('Y_att',Y_)
        return Y_

    def sequence_length(self, X, graph):
        """

        :param X: (batch size*max review length, max words num)
        :param graph: 
        :return: (batch size*max review length,)
        """
        paddings = tf.ones_like(X, dtype='int32') * self.nn_config['padding_word_index']
        condition = tf.equal(paddings, X)
        seq_len = tf.reduce_sum(tf.where(condition, tf.zeros_like(X, dtype='int32'), tf.ones_like(X, dtype='int32')),
                                axis=1, name='seq_len')
        return seq_len

    def review_length(self, X, graph):
        """
        
        :param X: shape=(batch size, max review length, wrods num)
        :param graph: 
        :return: 
        """
        paddings = tf.ones_like(X, dtype='int32') * self.nn_config['padding_word_index']
        condition = tf.equal(paddings, X)
        # shape=(batch size, max review length)
        condition = tf.reduce_all(condition,axis=2)
        tmp = tf.where(condition,tf.zeros_like(condition,dtype='float32'),tf.ones_like(condition,dtype='float32'))
        # shape = (batch size, )
        review_len = tf.reduce_sum(tmp,axis=1)
        return review_len

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
        mask = tf.reshape(mask, shape=[-1,self.nn_config['words_num']])
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
        keep_prob = tf.placeholder(dtype='float32')
        graph.add_to_collection('keep_prob_lstm', keep_prob)
        # graph.add_to_collection('keep_prob_lstm',keep_prob)
        cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size']),input_keep_prob=keep_prob , output_keep_prob=keep_prob,state_keep_prob=keep_prob)
        # outputs.shape = (batch size, max_time, cell size)
        outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=X, time_major=False, sequence_length=seq_len, dtype='float32')
        graph.add_to_collection('sentence_lstm_outputs', outputs)
        return outputs

    def sentence_bilstm(self,name, X, seq_len, reg, graph,scope_name=''):
        """
        return a lstm of a sentence
        :param X: shape = (batch size, words number, word dim)
        :param seq_len: shape = (batch size,) show the number of words in a batch
        :param graph: 
        :return: 
        """
        keep_prob_bilstm = tf.placeholder(dtype='float32')
        graph.add_to_collection('keep_prob_bilstm', keep_prob_bilstm)

        fw_cell = tf.nn.rnn_cell.LSTMCell(int(self.nn_config['lstm_cell_size'] / 2),)
        bw_cell = tf.nn.rnn_cell.LSTMCell(int(self.nn_config['lstm_cell_size'] / 2),)

        # outputs.shape = [(batch size, max time step, lstm cell size/2),(batch size, max time step, lstm cell size/2)]
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=X,sequence_length=seq_len,dtype='float32')
        # outputs.shape = (batch size, max time step, lstm cell size)
        outputs = tf.concat(outputs, axis=2, name='bilstm_outputs')
        reg[name].append(tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                    graph.get_tensor_by_name(scope_name+'/bidirectional_rnn/fw/lstm_cell/kernel:0')))
        reg[name].append(tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                    graph.get_tensor_by_name(scope_name+'/bidirectional_rnn/bw/lstm_cell/kernel:0')))
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

    def lookup_table(self, X, mask, table, graph):
        """
        :param X: shape = (batch_size, words numbers)
        :param mask: used to prevent update of #PAD#
        :return: shape = (batch_size, words numbers, word dim)
        """
        embeddings = tf.nn.embedding_lookup(table, X, partition_strategy='mod', name='lookup_table')
        embeddings = tf.multiply(embeddings, mask)
        return embeddings

    def sentiment_labels_input(self, graph):
        """
        :param graph: 
        :return: shape=[batch_size, number of attributes+1, 3], thus ys=[...,sentence[...,attj_senti[0,1,0],...],...]
        """
        Y_senti = tf.placeholder(shape=(None, self.nn_config['attributes_num']+1, self.nn_config['sentiment_num']),
                                 dtype='float32')
        # TODO: add non-attribute
        graph.add_to_collection('Y_senti', Y_senti)
        return Y_senti

    # ######################### #
    # CoarseSenti Net Version 2 #
    # ######################### #
    def context_matrix(self,reg,attr_sentence_repr):
        """

        :param reg:
        :param attr_sentence_repr: shape=(batch size*max review length, attributes num, n_layers*lstm cell size)
        :return:
        """
        shape = (self.nn_config['coarse_attributes_num'],self.nn_config['CoarseSenti_v2']['context_mat_size'],tf.shape(attr_sentence_repr)[-1])
        # shape = (attributes num, context num, sentence_repr dim)
        Z_mat = tf.get_variable(name='context_matrix',initializer=self.initializer(shape=shape,dtype='float32'))
        for key in reg:
            reg[key].append(tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(Z_mat))
        return Z_mat

    def document_attention(self,Z_mat, attr_sentence_repr, mask):
        """

        :param Z_mat: shape = (attributes num, context num, sentence_repr dim)
        :param attr_sentence_repr: shape = ( batch size*max review length, attributes num, n_layers*lstm cell size)
        :param mask: shape = (batch size * max review length, max words num)
        :return: attention of padded sentence is (0,0, ..., 0)
        """
        # TODO: need mask to eliminate influence of padded sentence
        # shape = (batch size * max review length,)
        mask = tf.reduce_max(mask,axis=1)
        # shape = (batch size, max review length)
        mask = tf.reshape(mask,shape=(-1,self.nn_config['max_review_len']))
        # shape = (batch size, context num, max review length)
        mask = tf.tile(tf.expand_dims(mask,axis=1),
                       multiples=[1,self.nn_config['CoarseSenti_v2']['context_mat_size'],1])

        # shape of each scalar: (1, context num, sentence_repr dim)
        Z_mat_ls = tf.split(Z_mat,num_or_size_splits=self.nn_config['coarse_attributes_num'],axis=0)
        # shape of each scalar: (batch size*max review length, 1, n_layers*lstm cell size)
        attr_sentence_repr_ls = tf.split(attr_sentence_repr,num_or_size_splits=self.nn_config['coarse_attributes_num'],axis=1)
        # shape = (attributes num, batch size*max review length, context num)
        document_attention_ls = []
        for z_mat,sentence_repr in zip(Z_mat_ls,attr_sentence_repr_ls):
            # shape = (context num, sentence repr dim)
            z_mat = tf.squeeze(z_mat)
            # shape = (batch size*max review length, n_layers*lstm cell size)
            sentence_repr = tf.squeeze(sentence_repr)
            # shape = (batch size*max review length, context num)
            d_atti = tf.matmul(sentence_repr,z_mat,transpose_b=True)
            # shape = (batch size, max review length, context num)
            d_atti = tf.reshape(d_atti,shape=(-1,self.nn_config['max_review_len'],
                                          self.nn_config['CoarseSenti_v2']['context_mat_size']))
            # shape = (batch size, context num, max review length)
            d_atti = tf.transpose(d_atti,perm=[0,2,1])
            # the padded sentence will be -inf
            d_atti = tf.add(d_atti,mask)

            # shape = (batch size, context num, max review length)
            document_attention = tf.nn.softmax(d_atti)
            # shape = (attributes num, batch size, context num, max review length)
            document_attention_ls.append(document_attention)
        return document_attention_ls


















