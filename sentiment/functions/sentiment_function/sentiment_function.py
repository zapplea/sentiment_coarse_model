import tensorflow as tf
import numpy as np


class SentiFunction:
    def __init__(self, nn_config):
        self.nn_config = nn_config

    # sentiment expression: for sentiment towards specific attribute
    # the sentiment is choosen by attribute
    def sentiment_matrix(self, graph):
        W = tf.get_variable(name='senti_mat', initializer=tf.random_uniform(shape=(
            self.nn_config['normal_senti_prototype_num'] * 3 + self.nn_config['attribute_senti_prototype_num'] *
            self.nn_config['attributes_num'],
            self.nn_config['sentiment_dim']), dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W))
        graph.add_to_collection('W', W)
        return W

    def sentiment_attention(self, H, W, m, graph):
        """
        :param h: shape = (batch size, number of words, lstm cell size)
        :param W: shape = (3*attribute numbers + 3,number of sentiment prototypes, lstm cell size). 3*attribute numbers is
        3 sentiment for each attributes; 3 is sentiment for non-attribute entity, it only has normal sentiment, not attribute
        specific sentiment.
        :param m: mask to eliminate influence of 0; (3*attributes number+3, number of sentiment expression prototypes)
        :return: shape = (batch size,number of words, 3+3*attributes number, number of sentiment prototypes).
        """
        # H.shape = (batch size, words num, 3+3*attributes number, word dim)
        H = tf.tile(tf.expand_dims(H, axis=2), multiples=[1, 1, 3 * self.nn_config['attributes_num'] + 3, 1])
        # H.shape = (batch size, words num, 3+3*attributes number, sentiment prototypes, word dim)
        H = tf.tile(tf.expand_dims(H, axis=3), multiples=[1, 1, 1, self.nn_config['normal_senti_prototype_num'] * 3 +
                                                          self.nn_config['attribute_senti_prototype_num'] *
                                                          self.nn_config['attributes_num'],
                                                          1])
        # temp.shape = (batch size, words num, 3+3*attributes number, sentiment prototypes num)
        temp = tf.multiply(m, tf.exp(tf.reduce_sum(tf.multiply(H, W), axis=4)))

        # denominator.shape = (batch size, words num, 3+3*attributes number, 1)
        denominator = tf.reduce_sum(temp, axis=3, keepdims=True)

        denominator = tf.tile(denominator, multiples=[1, 1, 1,
                                                      self.nn_config['normal_senti_prototype_num'] * 3 +
                                                      self.nn_config['attribute_senti_prototype_num'] * self.nn_config[
                                                          'attributes_num']])
        attention = tf.truediv(temp, denominator)
        graph.add_to_collection('senti_attention', attention)
        return attention

    def attended_sentiment(self, W, attention, graph):
        """
        :param W: all (yi,ai); shape = (3*number of attribute +3, sentiment prototypes, sentiment dim)
        :param attention: shape = (batch size,number of words, 3+3*attributes number, number of sentiment prototypes)
        :param graph: 
        :return: (batch size,number of words, 3+3*attributes number, sentiment dim)
        """
        # attention.shape = (batch size, number of words, 3+3*attributes number, number of sentiment prototypes, sentiment dim)
        attention = tf.tile(tf.expand_dims(attention, axis=4), multiples=[1, 1, 1, 1, self.nn_config['sentiment_dim']])
        attended_W = tf.reduce_sum(tf.multiply(attention, W), axis=3)
        graph.add_to_collection('attended_W', attended_W)
        return attended_W

    def item1(self, W, H, graph):
        """

        :param W: shape = (batch size,number of words, 3+3*attributes number, sentiment dim)
        :param H: shape = (batch size, number of words, word dim)
        :return: shape = (batch size,number of words, 3+3*attributes number)
        """
        # H.shape = (batch size,number of words, 3+3*attributes number, sentiment dim)
        H = tf.tile(tf.expand_dims(H, axis=2), multiples=[1, 1, 3 * self.nn_config['attributes_num'] + 3, 1])
        item1_score = tf.reduce_sum(tf.multiply(W, H), axis=3)
        graph.add_to_collection('item1_score', item1_score)
        return item1_score

    # association between attribute and sentiment: towards specific attribute
    def attribute_distribution(self, A, H, graph):
        """
        distribution of all attributes in this sentence
        :param A: A.shape = (attributes number +1 , attributes dim(=lstm cell size)) or 
                  A.shape = (batch size, number of words, number of attributes + 1, attribute dim(=lstm cell dim))
        :param H: batch size, words num, word dim
        :param graph: 
        :return: shape = (batch size, number of attributes+1, wrods number)
        """
        if not self.nn_config['is_mat']:
            H = tf.reshape(H, shape=(-1, self.nn_config['lstm_cell_size']))
            # A.shape=(number of attributes+1, attribute dim(=lstm cell size))
            # A_dist = (batch size,number of attributes+1,number of words)
            A_dist = tf.nn.softmax(tf.transpose(tf.reshape(tf.matmul(A, H, transpose_b=True),
                                                           shape=(self.nn_config['attributes_num'] + 1, -1,
                                                                  self.nn_config['words_num'])), [1, 0, 2]))
        else:
            # A.shape = (batch size, number of words, number of attributes+1, attribute dim(=lstm cell dim))
            # H.shape = (batch size, number of words, number of attributes+1, word dim)
            H = tf.tile(tf.expand_dims(H, axis=2), multiples=[1, 1, self.nn_config['attributes_num'] + 1, 1])
            # A_dist.shape = (batch size, attributes number, words number)
            A_dist = tf.nn.softmax(tf.transpose(tf.reduce_sum(tf.multiply(A, H), axis=3), [0, 2, 1]))

        graph.add_to_collection('attribute_distribution', A_dist)
        return A_dist

    def relative_pos_matrix(self, graph):
        V = tf.get_variable(name='relative_pos',
                            initializer=tf.random_uniform(shape=(self.nn_config['rps_num'], self.nn_config['rp_dim']),
                                                          dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(V))
        graph.add_to_collection('V', V)
        return V

    def relative_pos_ids(self, graph):
        """
        :param graph: 
        :return: shape = (number of words, number of words)
        """
        id4sentence = []
        for i in range(self.nn_config['words_num']):
            id4word_i = []
            for j in range(self.nn_config['words_num']):
                if abs(i - j) < self.nn_config['rps_num']:
                    id4word_i.append(abs(i - j))
                else:
                    id4word_i.append(self.nn_config['rps_num'] - 1)
            id4sentence.append(id4word_i)
        rp_ids = tf.constant(id4sentence, dtype='int32')
        graph.add_to_collection('relative_pos_ids', rp_ids)
        return rp_ids

    def rd_Vi(self, A_dist, V, rp_ids, graph):
        """
        :param A_dist: shape = (batch size, number of attributes+1, number of words)
        :param V: shape = (number of relative position, relative position dim)
        :param rp_ids: shape = (number of words, number of words)
        :param graph:
        :return: realtive position vector of each attribute at each position.
        shape = (batch size, number of attributes+1, number of words, relative position dim)
        """
        # rp_mat.shape = (number of words, number of words, rp_dim)
        rp_mat=tf.nn.embedding_lookup(V,rp_ids)
        # A_dist.shape = (batch size, number of attributes+1, number of words,relative position dim)
        A_dist = tf.tile(tf.expand_dims(A_dist,axis=3),multiples=[1,1,1,self.nn_config['rp_dim']])
        # A_dist.shape = (batch size, number of attributes+1, number of words, number of words,relative position dim)
        A_dist = tf.tile(tf.expand_dims(A_dist,axis=2),multiples=[1,1,self.nn_config['words_num'],1,1])
        # A_Vi.shape = (batch size, number of attributes+1, number of words, relative position dim)
        A_Vi = tf.reduce_sum(tf.multiply(A_dist,rp_mat),axis=3)
        graph.add_to_collection('A_Vi', A_Vi)
        return A_Vi

    def dp_Vi(self, A_dist, PD, graph):
        """

        :param A_dist: (batch size, number of attributes+1, wrods number)
        :param PD: (batch size, words num, words num, lstm cell size)
        :return: 
        """
        # PD.shape = (batch size, attributes num +1, words num, words num, lstm cell size)
        PD = tf.tile(tf.expand_dims(PD, axis=1), multiples=[1, self.nn_config['attributes_num'] + 1, 1, 1, 1])

        # A_dist.shape = (batch size, attributes num+1, words num, words num)
        A_dist = tf.tile(tf.expand_dims(A_dist, axis=2), multiples=[1, 1, self.nn_config['words_num'], 1])
        # A_dist.shape = (batch size, attributes num+1, words num, words num, lstm cell size)
        A_dist = tf.tile(tf.expand_dims(A_dist, axis=4), multiples=[1, 1, 1, 1, self.nn_config['lstm_cell_size']])
        # A_Vi.shape = (batch size, attributes num+1, words num, lstm cell size)
        A_Vi = tf.reduce_sum(tf.multiply(PD, A_dist), axis=3)
        graph.add_to_collection('A_Vi', A_Vi)
        return A_Vi

    def beta(self, graph):
        """

        :param graph: 
        :return: beta weight, shape=(rp_dim)
        """
        b = tf.get_variable(name='beta',
                            initializer=tf.random_uniform(shape=(self.nn_config['rp_dim'],), dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(b))
        graph.add_to_collection('beta', b)
        return b

    # sentiment score
    def score(self, item1, item2, mask, graph):
        """
        :param item1: shape = (batch size,number of words, 3+3*attributes number)
        :param item2: shape=(batch size, number of attributes+1, number of words)
        :param graph: 
        :return: (batch size, 3+3*attributes number, number of words) this is all combinations of yi and ai
        """
        # item1.shape = (batch size, 3+3*attributes number, number of words)
        item1 = tf.transpose(item1, [0, 2, 1])
        # item2.shape = (batch size, 3+3*attributes number, number of words)
        item2 = tf.reshape(tf.tile(tf.expand_dims(item2, axis=2), [1, 1, 3, 1]),
                           shape=(-1, 3 * self.nn_config['attributes_num'] + 3, self.nn_config['words_num']))
        # score.shape = (batch size, 3+3*attributes number, number of words)
        score = tf.add(item1, item2)
        # mask.shape = (batch size, attributes number, words num)
        mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, 3 + 3 * self.nn_config['attributes_num'], 1])
        # eliminate influence of #PAD#
        score = tf.add(score, mask)

        return score

    def max_false_senti_score(self, Y_senti, score, graph):
        """

        :param Y_senti: shape=(batch size, attributes numbers+1, 3)
        :param score: shape=(batch size, 3*attributes numbers+3)
        :param graph: 
        :return: shape = (batch size, number of attributes+1,3)
        """
        # score.shape = (batch size, attributes numbers+1, 3)
        score = tf.reshape(score, shape=(-1, self.nn_config['attributes_num'] + 1, 3))
        # if value is 1 then it is true, otherwise flase
        condition = tf.equal(tf.ones_like(Y_senti, dtype='float32'), Y_senti)
        # mask.shape = (batch size, attributes numbers+1, 3)
        mask = tf.where(condition,
                        tf.ones_like(score, dtype='float32') * tf.constant(-np.inf, dtype='float32'),
                        tf.zeros_like(score, dtype='float32'))
        # shape = (batch size, number of attributes+1,1)
        max_fscore = tf.reduce_max(tf.add(score, mask), axis=2, keepdims=True)

        # consider when attribute contains all sentiment in a sentence.
        max_fscore = tf.where(tf.is_inf(max_fscore), tf.zeros_like(max_fscore, dtype='float32'), max_fscore)
        # max_fscore.shape = (batch size, number of attributes+1, 3)
        max_fscore = tf.tile(max_fscore, multiples=[1, 1, 3])
        graph.add_to_collection('max_false_senti_score', max_fscore)
        return max_fscore

    def loss(self, Y_senti, score, max_false_score, graph):
        """
        :param Y_senti: shape=(batch size,attributes numbers+1, 3) the second part is one-hot to represent which sentiment it is.
        :param score: shape=(batch size, 3*number of attributes+3)
        :param max_false_score: shape = (batch size, number of attributes +1, 3)
        :param graph:
        :return: loss for a sentence for all true attributes and mask all false attributes.
        """
        # score.shape = (batch size, attribute number +1, 3)
        score = tf.reshape(score, shape=(-1, self.nn_config['attributes_num'] + 1, 3))
        theta = tf.constant(self.nn_config['sentiment_loss_theta'], dtype='float32')
        # senti_loss.shape = (batch size, attribute number +1, 3)
        senti_loss = tf.add(tf.subtract(theta, score), max_false_score)
        # masked_loss.shape = (batch size, attribute number +1, 3)
        masked_loss = tf.multiply(Y_senti, senti_loss)
        #
        zeros_loss = tf.zeros_like(masked_loss, dtype='float32')
        # masked_loss.shape = (batch size, attribute number +1, 3,1)
        zeros_loss = tf.expand_dims(zeros_loss, axis=3)
        masked_loss = tf.expand_dims(masked_loss, axis=3)
        loss = tf.reduce_max(tf.concat([zeros_loss, masked_loss], axis=3), axis=3)
        # TODO: eliminate the influence of batch size.
        batch_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(loss, axis=2), axis=1)) + tf.multiply(
            1 / self.nn_config['batch_size'], tf.reduce_sum(graph.get_collection('reg')))
        graph.add_to_collection('senti_loss', batch_loss)
        return batch_loss

    def prediction(self, score, Y_atr, graph):
        """
        :param score: shape = (batch size, attributes numbers+1,3)
        :param Y_atr: shape = (batch size, attributes numbers+1)
        :param graph: 
        :return: 
        """
        # score.shape = (batch size, attributes numbers+1,3)
        score = tf.nn.softmax(logits=score,axis=-1)
        # pred.shape =(batch size, attributes number +1)
        pred = tf.cast(tf.argmax(score,axis=2),dtype='float32')
        # use Y_atr to mask non-activated attributes' sentiment
        pred = tf.multiply(Y_atr, pred)

        graph.add_to_collection('prediction', pred)
        return pred

    def softmax_loss(self,labels, logits,graph):
        """
        
        :param labels: (batch size, number of attributes+1,3)
        :param logits: (batch_size, number of attributes + 1, 3)
        :return: 
        """

        loss = tf.reduce_mean(tf.add(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=logits,dim=-1),axis=1),
                                     tf.reduce_sum(graph.get_collection('reg'))))
        graph.add_to_collection('senti_loss', loss)
        return loss

    def accuracy(self, Y_senti, pred, graph):
        """

        :param Y_senti: shape = (batch size, attributes number +1,3)
        :param pred: shape = ()
        :param graph: 
        :return: 
        """
        condition = tf.equal(Y_senti, pred)
        cmp = tf.reduce_sum(
            tf.where(condition, tf.zeros_like(Y_senti, dtype='float32'), tf.ones_like(Y_senti, dtype='float32')),
            axis=2)
        condition = tf.equal(cmp, tf.zeros_like(cmp))
        accuracy = tf.reduce_mean(
            tf.where(condition, tf.ones_like(cmp, dtype='float32'), tf.zeros_like(cmp, dtype='float32')))
        graph.add_to_collection('accuracy', accuracy)
        return accuracy

    # ===============================================
    # ============= attribute function ==============
    # ===============================================
    def attribute_vec(self, graph):
        """

        :param graph: 
        :return: shape = (number of attributes+1, attributes dim)
        """
        # A is matrix of attribute vector
        A = tf.get_variable(name='A_vec', initializer=tf.random_uniform(
            shape=(self.nn_config['attributes_num'], self.nn_config['attribute_dim']),
            dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(A))
        graph.add_to_collection('A_vec', A)
        o = tf.get_variable(name='other_vec', initializer=tf.random_uniform(shape=(self.nn_config['attribute_dim'],),
                                                                            dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(o))
        graph.add_to_collection('o_vec', o)
        A = tf.concat([A, tf.expand_dims(o, axis=0)], axis=0)
        return A

    def attribute_mat(self, graph):
        """

        :param graph: 
        :return: shape = (attributes number+1, attribute mat size, attribute dim)
        """
        A_mat = tf.get_variable(name='A_mat', initializer=tf.random_uniform(shape=(self.nn_config['attributes_num'],
                                                                                   self.nn_config['attribute_mat_size'],
                                                                                   self.nn_config['attribute_dim']),
                                                                            dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(A_mat))
        graph.add_to_collection('A_mat', A_mat)
        o_mat = tf.get_variable(name='other_vec',
                                initializer=tf.random_uniform(shape=(self.nn_config['attribute_mat_size'],
                                                                     self.nn_config['attribute_dim']),
                                                              dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(o_mat))
        graph.add_to_collection('o_mat', o_mat)

        A_mat = tf.concat([A_mat, tf.expand_dims(o_mat, axis=0)], axis=0)

        return A_mat

    def words_attribute_mat2vec(self, H, A_mat, graph):
        """
        convert attribtes matrix to attributes vector for each words in a sentence. A_mat include non-attribute mention matrix.
        :param H: shape = (batch size, number of words, word dim)
        :param A_mat: (number of atr, atr mat size, atr dim)
        :param graph: 
        :return: shape = (batch size, number of words, number of attributes + 1, attribute dim(=lstm cell dim))
        """
        # H.shape = (batch size, words number, attribute number+1, word dim)
        H = tf.tile(tf.expand_dims(H, axis=2), multiples=[1, 1, self.nn_config['attributes_num'] + 1, 1])
        # H.shape = (batch size, words number, attribute number+1, attribute mat size, word dim)
        H = tf.tile(tf.expand_dims(H, axis=3), multiples=[1, 1, 1, self.nn_config['attribute_mat_size'], 1])
        # attention.shape = (batch size, words number, attribute number, attribute mat size)
        attention = tf.nn.softmax(tf.reduce_sum(tf.multiply(H, A_mat), axis=4))
        # attention.shape = (batch size, words number, attribute number, attribute mat size, attribute dim)
        attention = tf.tile(tf.expand_dims(attention, axis=4), multiples=[1, 1, 1, 1, self.nn_config['attribute_dim']])
        words_A = tf.reduce_sum(tf.multiply(attention, A_mat), axis=3)
        graph.add_to_collection('words_attributes', words_A)
        return words_A

    def sentences_input(self, graph):
        X = tf.placeholder(
            shape=(None, self.nn_config['words_num']),
            dtype='int32')
        graph.add_to_collection('X', X)
        return X

    def attribute_labels_input(self, graph):
        """

        :param graph: 
        :return: shape = (batch size, attributes number+1)
        """
        Y_att = tf.placeholder(shape=(None, self.nn_config['attributes_num']),
                               dtype='float32')
        # TODO: add non-attribute
        batch_size = tf.shape(Y_att)[0]
        non_attr = tf.zeros((batch_size,1),dtype='float32')
        condition = tf.equal(tf.reduce_sum(Y_att,axis=1,keepdims=True),non_attr)
        non_attr = tf.where(condition,non_attr,tf.ones_like(non_attr))
        Y_att = tf.concat([Y_att,non_attr],axis=1)

        graph.add_to_collection('Y_att', Y_att)
        return Y_att

    def sentiment_labels_input(self, graph):
        """
        :param graph: 
        :return: shape=[batch_size, number of attributes+1, 3], thus ys=[...,sentence[...,attj_senti[0,1,0],...],...]
        """
        Y_senti = tf.placeholder(shape=(None, self.nn_config['attributes_num']+1, 3),
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
        mask = tf.where(condition, tf.ones_like(X, dtype='float32') * tf.convert_to_tensor(-np.inf),
                        tf.zeros_like(X, dtype='float32'))
        return mask

    def sentence_lstm(self, X, seq_len, graph):
        """
        return a lstm of a sentence
        :param X: shape = (batch size, words number, word dim)
        :param seq_len: shape = (batch size,) show the number of words in a batch
        :param graph: 
        :return: 
        """
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size'])
        # outputs.shape = (batch size, max_time, cell size)
        outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=X, time_major=False, sequence_length=seq_len, dtype='float32')
        graph.add_to_collection('sentence_lstm_outputs', outputs)
        graph.add_to_collection('reg',
                                tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                    graph.get_tensor_by_name('sentence_lstm/rnn/basic_lstm_cell/kernel:0')))
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
        tf.add_to_collection('keep_prob_lstm',keep_prob)
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

    def path_dependency_table_input(self, graph):
        """
        :param graph: 
        :return: 
        """
        PD = tf.placeholder(shape=(None,
                                   self.nn_config['words_num'],
                                   self.nn_config['words_num'],
                                   self.nn_config['max_path_length']),
                            dtype='int32')
        graph.add_to_collection('path_dependency', PD)
        return PD

    def path_sequence_length(self, PD):
        """
        The PD is words id
        :param PD: matrix of path dependency (batch size, words num, words num, max path length): (sentences number, 
                                                                                            words num, 
                                                                                            tables number for each word, 
                                                                                            length of path)
        :return: (batch size, words num, words num)
        """
        PD = tf.reshape(PD, shape=(-1, self.nn_config['max_path_length']))
        paddings = tf.ones_like(PD, dtype='int32') * self.nn_config['padding_word_index']
        condition = tf.equal(paddings, PD)
        seq_len = tf.reduce_sum(tf.where(condition, tf.zeros_like(PD, dtype='int32'), tf.ones_like(PD, dtype='int32')),
                                axis=1, name='path_seq_len')
        return seq_len

    def path_dependency_bilstm(self, PD, seq_len, graph):
        """

        :param PD: matrix of path dependency (batch size, words num, words num, max path length, word dim): (sentences number, 
                                                                                                              words num, 
                                                                                                              tables number for each word, 
                                                                                                              words num for each row in a table,
                                                                                                              word dim)
        :param seq_len: (batch size*words num*words num, )
        :param graph: 
        :return: 
        """
        current_batch_size = tf.shape(PD)[0]
        PD = tf.reshape(PD, shape=(-1, self.nn_config['max_path_length'], self.nn_config['word_dim']))
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(int(self.nn_config['lstm_cell_size']/2))
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(int(self.nn_config['lstm_cell_size']/2))
        # outputs.shape = [(-1,max words num, lstm cell size/2),(-1,max words num, lstm cell size/2)]
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                     cell_bw=bw_cell,
                                                     inputs=PD,
                                                     sequence_length=seq_len,
                                                     time_major=False,dtype='float32')
        # outputs.shape = (batch size*words num*words num, max path length, word dim)
        outputs = tf.concat(outputs, axis=2, name='bilstm_outputs')
        # instance_index stands for the index of dependency path
        instance_index = tf.cast(tf.expand_dims(tf.range(start=0, limit=current_batch_size * self.nn_config[
            'words_num'] * self.nn_config['words_num']),
                                                axis=1), dtype='int32')
        # instance_length_index stands for the length of the instance
        # for padded sentence, the seq length is 0; then the index will be -1 and still the [0,...,0] will be chosen.
        instance_length_index = tf.cast(tf.expand_dims(seq_len - 1, axis=1), dtype='int32')
        slice_index = tf.concat([instance_index, instance_length_index], axis=1)
        outputs = tf.gather_nd(outputs, slice_index)
        print(outputs)
        outputs = tf.reshape(outputs, shape=(-1,
                                             self.nn_config['words_num'],
                                             self.nn_config['words_num'],
                                             self.nn_config['lstm_cell_size']))
        graph.add_to_collection('reg',
                                tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                    graph.get_tensor_by_name(
                                        'dependency_path_bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')))
        graph.add_to_collection('reg',
                                tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                    graph.get_tensor_by_name(
                                        'dependency_path_bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')))
        return outputs

    def lstm_mask(self, X):
        # need to pad the #PAD# words to zeros, otherwise, they will be junks.
        X = tf.cast(X, dtype='float32')
        ones = tf.ones_like(X, dtype='float32') * self.nn_config['padding_word_index']
        is_one = tf.equal(X, ones)
        mask = tf.where(is_one, tf.zeros_like(X, dtype='float32'), tf.ones_like(X, dtype='float32'))
        mask = tf.tile(tf.expand_dims(mask, axis=2), multiples=[1, 1, self.nn_config['lstm_cell_size']])
        return mask

    def senti_extors_mat(self, graph):
        """
        input a matrix to extract sentiment expression for attributes in sentences. The last one extract sentiment expression for non-attribute.
        The non-attribute only has one sentiment: NEU
        shape of the extractors matrix: [3*attribute numbers+1, 
                              normal sentiment prototype numbers*3 + attributes sentiment prototypes number*attribute number,
                              sentiment epression dim(sentiment dim)]
        :param graph: 
        :return: 
        """
        extors = self.sentiment_extract_mat()
        extors = tf.constant(extors, dtype='float32')
        graph.add_to_collection('senti_extractor', extors)
        return extors

    # @ normal_function
    def extor_expandNtile(self, extor, proto_num):
        """

        :param exter: it is extractor  
        :return: 
        """
        extor = np.expand_dims(extor, axis=1)
        extor = np.tile(extor, reps=[1, self.nn_config[proto_num]])
        extor = np.expand_dims(extor, axis=2)
        extor = np.tile(extor, reps=[1, 1, self.nn_config['sentiment_dim']])
        extor = np.reshape(extor, (-1, self.nn_config['sentiment_dim']))
        return extor

    # @ normal_function
    def sentiment_extract_mat(self):
        """
        This function return a matrix to extract expression prototype for all (yi,ai) combinations.
        :return: [ sentiment extractor for one sentence[...,[1,...],...,[0,...],...] ,...] 
        """
        # label = np.ones(shape=(self.nn_config['attributes_num'],), dtype='float32')
        extors = []
        for i in range(self.nn_config['attributes_num']):
            # attribute sentiment prototypes
            att_senti_ext = np.zeros(shape=(self.nn_config['attributes_num'],), dtype='float32')
            att_senti_ext[i] = 1
            att_senti_ext = self.extor_expandNtile(att_senti_ext, proto_num='attribute_senti_prototype_num')
            for j in range(3):
                # normal sentiment prototypes
                normal_senti_ext = np.zeros(shape=(3,), dtype='float32')
                normal_senti_ext[j] = 1
                normal_senti_ext = self.extor_expandNtile(normal_senti_ext, proto_num='normal_senti_prototype_num')
                extors.append(np.concatenate([normal_senti_ext, att_senti_ext], axis=0))
        for i in range(3):
            # non-attribute sentiment prototypes
            o_att_senti_ext = np.zeros(shape=(self.nn_config['attributes_num']), dtype='float32')
            o_att_senti_ext = self.extor_expandNtile(o_att_senti_ext, 'attribute_senti_prototype_num')
            # non-attribute normal sentiment prototypes
            o_normal_senti_ext = np.zeros(shape=(3,), dtype='float32')
            o_normal_senti_ext[i] = 1
            o_normal_senti_ext = self.extor_expandNtile(o_normal_senti_ext, proto_num='normal_senti_prototype_num')
            extors.append(np.concatenate([o_normal_senti_ext, o_att_senti_ext], axis=0))
        return np.array(extors)

    def extors_mask(self, extors, graph):
        """
        when calculate p(w|h), need to eliminate the the influence of false sentiment.
        :param extor: shape = (3*attribute numbers +3, 
                               self.nn_config['normal_senti_prototype_num'] * 3 +
                               self.nn_config['attribute_senti_prototype_num'] * self.nn_config['attributes_num'],
                               sentiment dim)
        :param graph: 
        :return: (3*attributes number+3, number of sentiment expression prototypes)
        """
        extors = tf.reduce_sum(extors, axis=2)
        condition = tf.equal(extors, np.zeros_like(extors, dtype='float32'))
        mask = tf.where(condition, tf.zeros_like(extors, dtype='float32'), tf.ones_like(extors, dtype='float32'))
        graph.add_to_collection('extors_mask', mask)
        return mask

    def optimizer(self, loss, graph):
        opt = tf.train.AdamOptimizer(self.nn_config['lr']).minimize(loss)
        graph.add_to_collection('opt', opt)
        return opt

    def wordEbmedding_table_input(self,graph):
        table = tf.placeholder(shape=(self.nn_config['lookup_table_words_num'], self.nn_config['word_dim']),
                               dtype='float32')
        graph.add_to_collection('table', table)
        rel_word_embeddings = tf.random_uniform(shape=(self.nn_config['rel_words_num'], self.nn_config['rel_word_dim']),
                                                dtype='float32')
        table = tf.concat([table, rel_word_embeddings], axis=0)
        table = tf.Variable(table, name='table')
        return table

    def lookup_table(self, X, mask, table, graph):
        """
        :param X: shape = (batch_size, words numbers)
        :param mask: used to prevent update of #PAD#
        :return: shape = (batch_size, words numbers, word dim)
        """
        embeddings = tf.nn.embedding_lookup(table, X, partition_strategy='mod', name='lookup_table')
        embeddings = tf.multiply(embeddings, mask)
        graph.add_to_collection('lookup_table', embeddings)
        return embeddings

    def is_word_padding_input(self, ids, graph):
        """
        To make the sentence have the same length, we need to pad each sentence with '#PAD#'. To avoid updating of the vector,
        we need a mask to multiply the result of lookup table.
        :param graph: 
        :return: 
        """
        ids = tf.cast(ids, dtype='float32')
        ones = tf.ones_like(ids, dtype='float32') * self.nn_config['padding_word_index']
        is_pad = tf.equal(ids, ones)
        mask = tf.where(is_pad, tf.zeros_like(ids, dtype='float32'), tf.ones_like(ids, dtype='float32'))
        mask = tf.tile(tf.expand_dims(mask, axis=2), multiples=[1, 1, self.nn_config['word_dim']])
        return mask

    def is_dpword_padding_input(self,ids,graph):
        """
        
        :param ids: (batch size, words number, words number, max dp length)
        :param graph: 
        :return: 
        """
        ids = tf.cast(ids, dtype='float32')
        ones = tf.ones_like(ids, dtype='float32') * self.nn_config['padding_word_index']
        is_pad = tf.equal(ids, ones)
        mask = tf.where(is_pad, tf.zeros_like(ids, dtype='float32'), tf.ones_like(ids, dtype='float32'))
        mask = tf.tile(tf.expand_dims(mask,axis=4),multiples=[1,1,1,1,self.nn_config['rel_word_dim']])
        return mask