import tensorflow as tf
import numpy as np


class AttributeFunction:
    def __init__(self, nn_config):
        self.nn_config = nn_config

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
        o = tf.get_variable(name='other_vec', initializer=tf.random_uniform(shape=(1, self.nn_config['attribute_dim']),
                                                                            dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(o))
        graph.add_to_collection('o_vec', o)
        return A, o

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
                                initializer=tf.random_uniform(shape=(1,
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
        # H.shape = (batch size, words number, attribute number, word dim)
        H = tf.tile(tf.expand_dims(H, axis=2), multiples=[1, 1, self.nn_config['attributes_num'], 1])
        # H.shape = (batch size, words number, attribute number+1, attribute mat size, word dim)
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

    def score(self, A, X, graph):
        """

        :param A: shape = (number of attributes, attribute dim) or
                  shape = (batch size, words number, attributes num, attribute dim)
        :param X: shape = (batch size, words number, lstm cell size)
        :param graph: 
        :return: (batch size, attributes num)
        """
        if not self.nn_config['is_mat']:
            X = tf.reshape(X, shape=(-1, self.nn_config['lstm_cell_size']))
            # score.shape = (attributes num,batch size*words num)
            score = tf.matmul(A, X, transpose_b=True)
            # score.shape = (attributes num, batch size, words num)
            score = tf.reshape(score, (self.nn_config['attributes_num'], -1, self.nn_config['words_num']))
            # score.shape = (batch size, attributes number, words num)
            score = tf.transpose(score, [1, 0, 2])
            # score.shape = (batch size, attributes num)
            score = tf.reduce_max(score, axis=2)
        else:
            # X.shape = (batch size, words num, attributes num, attribute dim)
            X = tf.tile(tf.expand_dims(X, axis=2), multiples=[1, 1, self.nn_config['attributes_num'], 1])
            # score.shape = (batch size, words num, attributes num)
            score = tf.reduce_sum(tf.multiply(A, X), axis=3)
            # score.shape = (batch size, attributes num, words num)
            score = tf.transpose(score, [0, 2, 1])
            # score.shape = (batch size, attributes num)
            score = tf.reduce_max(score, axis=2)
        graph.add_to_collection('score', score)
        return score

    def prediction(self, score, graph):
        condition = tf.greater(score, tf.ones_like(score, dtype='float32') * self.nn_config['atr_threshold'])
        pred = tf.where(condition, tf.ones_like(score, dtype='float32'), tf.zeros_like(score, dtype='float32'))
        graph.add_to_collection('atr_pred', pred)
        return pred

    def accuracy(self, Y_att, pred, graph):
        """

        :param Y_att: shape = (batch size, attributes number)
        :param pred: shape = (batch size, attributes number)
        :param graph: 
        :return: 
        """
        condition = tf.equal(Y_att, pred)
        cmp = tf.reduce_sum(
            tf.where(condition, tf.zeros_like(Y_att, dtype='float32'), tf.ones_like(Y_att, dtype='float32')), axis=1)
        condition = tf.equal(cmp, tf.zeros_like(cmp))
        accuracy = tf.reduce_mean(
            tf.where(condition, tf.ones_like(cmp, dtype='float32'), tf.zeros_like(cmp, dtype='float32')))
        graph.add_to_collection('accuracy', accuracy)
        return accuracy

    def max_false_score(self, score, Y_att, graph):
        """

        :param score: shape = (batch size, attributes num)
        :param Y_att: shape = (batch size, attributes num)
        :param graph:
        :return: 
        """
        condition = tf.equal(Y_att, tf.ones_like(Y_att, dtype='float32'))
        max_fscore = tf.reduce_max(
            tf.where(condition, tf.ones_like(score) * tf.constant(-np.inf, dtype='float32'), score), axis=1,
            keep_dims=True)
        max_fscore = tf.where(tf.is_inf(max_fscore), tf.zeros_like(max_fscore, dtype='float32'), max_fscore)
        max_fscore = tf.tile(max_fscore, multiples=[1, self.nn_config['attributes_num']])
        graph.add_to_collection('max_false_score', max_fscore)
        return max_fscore

    def loss(self, score, max_fscore, Y_att, graph):
        """

        :param score: shape = (batch size, attributes num)
        :param max_fscore: shape = (batch size, attributes num)
        :param Y_att: (batch size, attributes num)
        :param graph: 
        :return: (batch size, attributes number)
        """
        # create a mask for non-attribute in which Sy is 0 and need to keep max false attribute
        condition = tf.equal(tf.reduce_sum(Y_att, axis=1),
                             tf.zeros(shape=(self.nn_config['batch_size'],), dtype='float32'))
        item1 = np.zeros(shape=(self.nn_config['batch_size'], self.nn_config['attributes_num']), dtype='float32')
        for i in range(self.nn_config['batch_size']):
            item1[i][0] = 1
        nonatr_mask = tf.where(condition, tf.constant(item1, dtype='float32'), Y_att)
        #
        theta = tf.constant(self.nn_config['attribute_loss_theta'], dtype='float32')
        # loss.shape = (batch size, attributes num)
        loss = tf.multiply(tf.add(tf.subtract(theta, tf.multiply(Y_att, score)), max_fscore), nonatr_mask)
        zero_loss = tf.zeros_like(loss, dtype='float32')

        loss = tf.expand_dims(loss, axis=2)
        zero_loss = tf.expand_dims(zero_loss, axis=2)
        # loss.shape = (batch size, attributes num)
        loss = tf.reduce_max(tf.concat([loss, zero_loss], axis=2), axis=2)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1)) + tf.multiply(1 / self.nn_config['batch_size'],
                                                                         tf.reduce_sum(graph.get_collection('reg')))
        graph.add_to_collection('atr_loss', loss)

        return loss


class Classifier:
    def __init__(self, nn_config, data_generator):
        self.nn_config = nn_config
        self.dg = data_generator
        self.af = AttributeFunction(nn_config)

    def sentences_input(self, graph):
        X = tf.placeholder(
            shape=(self.nn_config['batch_size'], self.nn_config['words_num']),
            dtype='float32')
        graph.add_to_collection('X', X)
        return X

    def attribute_labels_input(self, graph):
        Y_att = tf.placeholder(shape=(self.nn_config['batch_size'], self.nn_config['attributes_num']), dtype='float32')
        graph.add_to_collection('Y_att', Y_att)
        return Y_att

    # should use variable share
    def sentence_lstm(self, X, mask, graph):
        """
        return a lstm of a sentence
        :param X: shape = (batch size, words number, word dim)
        :param mask: shape = (batch size, words number, lstm cell size)
        :param graph: 
        :return: 
        """
        weight = tf.get_variable(name='sentence_lstm_w',
                                 initializer=tf.random_uniform(shape=(self.nn_config['word_dim'],
                                                                      self.nn_config['lstm_cell_size']),
                                                               dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(weight))
        bias = tf.get_variable(name='sentence_lstm_b',
                               initializer=tf.zeros(shape=(self.nn_config['lstm_cell_size']), dtype='float32'))

        X = tf.reshape(X, shape=(-1, self.nn_config['word_dim']))
        Xt = tf.add(tf.matmul(X, weight), bias)
        Xt = tf.reshape(Xt, shape=(-1, self.nn_config['words_num'], self.nn_config['lstm_cell_size']))
        # xt = tf.add(tf.expand_dims(tf.matmul(x, weight), axis=0), bias)
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size'])
        init_state = cell.zero_state(batch_size=self.nn_config['batch_size'], dtype='float32')
        # outputs.shape = (batch size, max_time, cell size)
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs=Xt, initial_state=init_state, time_major=False)
        outputs = tf.multiply(outputs, mask)
        graph.add_to_collection('sentence_lstm_outputs', outputs)
        return outputs

    def lstm_mask(self, X):
        # need to pad the #PAD# words to zeros, otherwise, they will be junks.
        X = tf.cast(X, dtype='float32')
        ones = tf.ones_like(X, dtype='float32') * self.nn_config['padding_word_index']
        is_one = tf.equal(X, ones)
        mask = tf.where(is_one, tf.zeros_like(X, dtype='float32'), tf.ones_like(X, dtype='float32'))
        mask = tf.tile(tf.expand_dims(mask, axis=2), multiples=[1, 1, self.nn_config['lstm_cell_size']])
        return mask

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

    def classifier(self):
        graph = tf.Graph()
        with graph.as_default():
            X = self.sentences_input(graph=graph)
            words_pad_M = self.is_word_padding_input(X, graph)
            lstm_mask = self.lstm_mask(X)
            X = self.lookup_table(X, words_pad_M, graph)
            # lstm
            with tf.variable_scope('sentence_lstm'):
                # H.shape = (batch size, max_time, cell size)
                H = self.sentence_lstm(X, lstm_mask, graph=graph)
                graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                    graph.get_tensor_by_name('sentence_lstm/rnn/basic_lstm_cell/kernel:0')))
            # Y_att.shape = (batch size, number of attributes+1)
            Y_att = self.attribute_labels_input(graph=graph)
            if not self.nn_config['is_mat']:
                A, o = self.af.attribute_vec(graph)
                A = A - o
            else:
                A, o = self.af.attribute_mat(graph)
                # A.shape = (batch size, words num, attributes number, attribute dim)
                A_lstm = self.af.words_attribute_mat2vec(H, A, graph)
                o_lstm = self.af.words_nonattribute_mat2vec(H, o, graph)
                A_lstm = A_lstm - o_lstm
                A_e = self.af.words_attribute_mat2vec(X,A,graph)
                o_e = self.af.words_nonattribute_mat2vec(X,A,graph)
                A_e = A_e-o_e
            if not self.nn_config['is_mat']:
                score_lstm = self.af.score(A, H, graph)
                score_e = self.af.score(A,X,graph)
                score = tf.add(score_lstm,score_e)
            else:
                score_lstm = self.af.score(A_lstm,H,graph)
                score_e = self.af.score(A_e,X,graph)
                score = tf.add(score_lstm, score_e)

            max_fscore = self.af.max_false_score(score, Y_att, graph)
            loss = self.af.loss(score, max_fscore, Y_att, graph)
            pred = self.af.prediction(score, graph)
            accuracy = self.af.accuracy(Y_att, pred, graph)
        with graph.as_default():
            opt = self.optimizer(loss, graph=graph)
            saver = tf.train.Saver()
        return graph, saver

    def train(self):
        graph, saver = self.classifier()
        with graph.as_default():
            # input
            X = graph.get_collection('X')[0]
            # labels
            Y_att = graph.get_collection('Y_att')[0]
            # train_step
            train_step = graph.get_collection('train_step')[0]
            #
            table = graph.get_collection('table')[0]
            #
            accuracy = graph.get_collection('accuracy')[0]
            #
            loss = graph.get_collection('atr_loss')[0]
            # attribute function
            init = tf.global_variables_initializer()
        table_data = self.dg.table_generator()
        with graph.device('/gpu:1'):
            with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                sess.run(init, feed_dict={table: table_data})
                for i in range(self.nn_config['epoch']):
                    sentences, Y_att_data = self.dg.data_generator('train', i)
                    sess.run(train_step, feed_dict={X: sentences, Y_att: Y_att_data})

                    if i % 5000 == 0 and i != 0:
                        sentences, Y_att_data = self.dg.data_generator('test')
                        valid_size = Y_att_data.shape[0]
                        p = 0
                        l = 0
                        count = 0
                        batch_size = self.nn_config['batch_size']
                        for i in range(valid_size // batch_size):
                            count += 1
                            p += sess.run(accuracy, feed_dict={X: sentences[i * batch_size:i * batch_size + batch_size],
                                                               Y_att: Y_att_data[
                                                                      i * batch_size:i * batch_size + batch_size]})
                            l += sess.run(loss, feed_dict={X: sentences[i * batch_size:i * batch_size + batch_size],
                                                           Y_att: Y_att_data[
                                                                  i * batch_size:i * batch_size + batch_size]})
                        p = p / count
                        l = l / count