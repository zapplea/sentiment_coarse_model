import tensorflow as tf
import numpy as np
from sentiment.util.coarse.atr_data_generator import DataGenerator
from sentiment.util.coarse.metrics import Metrics


class AttributeFunction:
    def __init__(self, nn_config):
        self.nn_config = nn_config

    def attribute_vec(self, graph):
        # A is matrix of attribute vector
        A = []
        for i in range(self.nn_config['attributes_num']):
            att_vec = tf.get_variable(name='att_vec' + str(i),
                                      initializer=tf.random_uniform(shape=(self.nn_config['attribute_dim'],),
                                                                    dtype='float32'))
            graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(att_vec))
            A.append(att_vec)
        graph.add_to_collection('A', A)
        o = tf.get_variable(name='other_vec', initializer=tf.random_uniform(shape=(self.nn_config['attribute_dim'],),
                                                                            dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(o))
        graph.add_to_collection('o', o)
        return A, o

    def score(self, A_mat, o_mat, x, graph):
        """
        :param A_mat: 
        :param o_mat: 
        :param x: a sentence 
        :param graph: 
        :return: [att1_score,att2_score,...,attk_score]; attj_score is a scalar.
        """
        A = tf.subtract(A_mat, o_mat)
        result = tf.reduce_max(tf.matmul(A, x, transpose_b=True), axis=1)
        graph.add_to_collection('att_score', result)
        return result

    def prediction(self, sentences_atr_score, sentences_relevance_weight, graph):
        """
        Pay attention to padded sentences: for padded sentences, the p(x|a,y) = 0
        :param sentences_atr_score: shape = (review batch size * sentences number, attributes number)
        :param sentences_relevance_weight: (review batch size, number of sentences, number of attributes). attribute labels for each review
        :param ispad: shape = (review batch size * sentences number, )   one hot vector to show which sentence is padding
        :param graph: 
        :return: shape = (batch size, number of attributes)
        """
        sentences_atr_score = tf.reshape(sentences_atr_score, shape=(self.nn_config['batch_size'],
                                                                     self.nn_config['sentences_num'],
                                                                     self.nn_config['attributes_num']))
        # weighted_atr_score.shape = (batch size, number of sentences, number of attributes)
        weighted_atr_score = tf.multiply(sentences_atr_score, sentences_relevance_weight)
        # reviews_atr_score = (batch size, number of attributes)
        reviews_atr_score = tf.reduce_sum(weighted_atr_score, axis=1)
        condition = tf.greater(reviews_atr_score,
                               tf.ones_like(reviews_atr_score, dtype='float32') * self.nn_config['atr_score_threshold'])
        pred = tf.where(condition,
                        tf.ones_like(reviews_atr_score, dtype='float32'),
                        tf.zeros_like(reviews_atr_score, dtype='float32'))
        graph.add_to_collection('reviews_atr_pred', pred)
        return pred

    def max_f_score(self, score, atr_label):
        condition = tf.equal(tf.ones_like(atr_label, dtype='float32'), atr_label)
        max_fscore = tf.reduce_max(tf.where(condition,
                                            tf.ones_like(atr_label, dtype='float32') * tf.constant(-np.inf,
                                                                                                   dtype='float32'),
                                            score))
        # consider when a sentence contains all attributes
        max_fscore = tf.where(tf.is_inf(max_fscore), tf.zeros_like(max_fscore, dtype='float32'), max_fscore)
        return max_fscore

    def loss(self, score, atr_label, sentence_relevance_weight, graph):
        """

        :param score: 
        :param pred: 
        :param atr_label: shape = (number of attributes,), each scalar is p(a|D)
        :param sentence_relevance_weight: shape = (number of attributes, ), each scalar is p(x|a,y)
        :param graph: 
        :return: 
        """
        # max wrong score
        # extract the max false attributes score
        max_fscore = self.max_f_score(score, atr_label)

        theta = tf.constant(self.nn_config['attribute_loss_theta'], dtype='float32')

        # use attribute threshold to process attribute probability vector
        condition = tf.greater(atr_label, self.nn_config['label_atr_threshold'])
        atr_label_1hot = tf.where(condition, tf.ones_like(atr_label, dtype='float32'),
                                  tf.zeros_like(atr_label, dtype='float32'))
        atr_label_thresheld = tf.where(condition, atr_label,
                                       tf.zeros_like(atr_label, dtype='float32'))
        loss = tf.add(tf.subtract(theta, tf.multiply(atr_label_1hot, score)), max_fscore)
        # p(a|D)
        # non-attribute and attribute uses different mask
        condition = tf.equal(tf.reduce_sum(atr_label_thresheld), 0)
        nonatr_label = np.zeros(shape=(self.nn_config['attributes_num'],), dtype='float32')
        nonatr_label[0] = 1
        nonatr_label = tf.constant(nonatr_label, dtype='float32')
        mask = tf.cond(condition, lambda: nonatr_label, lambda: atr_label_thresheld)
        # p(x|a,y)
        # non-attribute and attributes use different sentence relevance weight. for non-attribute situation, p(x|o) should be 1
        nonatr_relevance_weight = np.zeros(shape=(self.nn_config['attributes_num'],), dtype='float32')
        nonatr_relevance_weight[0] = 1
        nonatr_relevance_weight = tf.constant(nonatr_relevance_weight, dtype='float32')
        relevance_weight = tf.cond(condition, lambda: nonatr_relevance_weight, lambda: sentence_relevance_weight)

        # mask loss so that the false label will not be updated.
        masked_loss = tf.multiply(mask, tf.multiply(relevance_weight, loss))
        # max{0,loss}
        zero_item = tf.zeros_like(atr_label, dtype='float32')
        zero_item = tf.expand_dims(zero_item, axis=1)
        loss = tf.expand_dims(masked_loss, axis=1)
        loss = tf.reduce_max(tf.concat([zero_item, loss], axis=1), axis=1)
        loss = tf.reduce_sum(loss)
        graph.add_to_collection('atr_loss', loss)
        return loss


class Classifier:
    def __init__(self, nn_config):
        self.nn_config = nn_config
        self.dg = DataGenerator(nn_config)
        self.af = AttributeFunction(nn_config)

    def reviews_input(self, graph):
        """
        This function feed review as training data for the model
        :param graph: 
        :return: shape = (review batch size, number of sentences, word dim)
        """
        R = tf.placeholder(
            shape=(self.nn_config['batch_size'], self.nn_config['sentences_num'], self.nn_config['words_num']),
            dtype='int32')
        graph.add_to_collection('R', R)
        return R

    def is_sentence_padding_input(self, graph):
        """
        one hot matrix. 0 represent the sentence is padded.
        This function input a mask. When calculate p(x|a,y) and loss, it eliminate influence of padded sentences. This mask is calculated outside the model.
        for padding sentence, it is zeros at that position.
        :param graph: 
        :return: 
        """
        mask = tf.placeholder(shape=(self.nn_config['batch_size'], self.nn_config['sentences_num']), dtype='float32')
        graph.add_to_collection('sentences_padding', mask)
        return mask

    def attribute_labels_input(self, graph):
        """
        each review batch have a attribute label. each scalar in attribute vector represent the possibility of the attributes
        :param graph: 
        :return: 
        """
        y_att = tf.placeholder(shape=(self.nn_config['batch_size'], self.nn_config['attributes_num']), dtype='float32')
        graph.add_to_collection('y_att', y_att)
        return y_att

    def Sr(self, Hl, A, graph):
        """
        :param Hl: shape = (number of sentences, lstm cell size)
        :param A: shape = (number of attributes, attribute dim)
        :param graph: 
        :return: shape = (number of sentences, number of attribute)
        """
        Wa = tf.get_variable(name='Sr_Wa', initializer=tf.random_uniform(
            shape=(self.nn_config['lstm_cell_size'], self.nn_config['attribute_dim']),
            dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(Wa))
        # Sr.shape = (number of sentences, number of attributes)
        relevance_score = tf.matmul(tf.matmul(Hl, Wa), A, transpose_b=True)
        graph.add_to_collection('Sr', relevance_score)
        return relevance_score

    def sentence_weight(self, relevance_score, mask, graph):
        """

        :param relevance_score: shape = (sentences numbers, attributes numbers)
        :param mask: shape = (sentences numbers, )
        :param graph: 
        :return: (attributes numbers, sentences numbers)
        """
        # score.shape = (attributes numbers, sentences numbers)
        score = tf.transpose(relevance_score)
        exp_score = tf.multiply(tf.exp(score), mask)
        relevance_weight = tf.divide(exp_score, tf.tile(tf.reduce_sum(exp_score, axis=1, keep_dims=True),
                                                        multiples=[1, self.nn_config['sentences_num']]))
        graph.add_to_collection('sentence_relevance_weight', relevance_weight)
        return relevance_weight

    # should use variable share
    def sentence_lstm(self, X, reuse, graph):
        """
        return a lstm of a sentence
        :param X: sentences in a review
        :param graph: 
        :return: 
        """
        weight = tf.get_variable(name='sentence_lstm_w',
                                 initializer=tf.random_uniform(shape=(self.nn_config['word_dim'],
                                                                      self.nn_config['lstm_cell_size']),
                                                               dtype='float32'))
        if reuse == None:
            graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(weight))
        bias = tf.get_variable(name='sentence_lstm_b',
                               initializer=tf.zeros(shape=(self.nn_config['lstm_cell_size']), dtype='float32'))

        X = tf.reshape(X, shape=(-1, self.nn_config['word_dim']))
        Xt = tf.add(tf.matmul(X, weight), bias)
        Xt = tf.reshape(Xt, shape=(-1, self.nn_config['words_num'], self.nn_config['lstm_cell_size']))
        # xt = tf.add(tf.expand_dims(tf.matmul(x, weight), axis=0), bias)
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size'])
        init_state = cell.zero_state(batch_size=self.nn_config['sentences_num'], dtype='float32')
        # outputs.shape = (batch size, max_time, cell size)
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs=Xt, initial_state=init_state, time_major=False)
        graph.add_to_collection('sentence_lstm_outputs', outputs)
        return outputs

    def optimizer(self, joint_losses, graph):
        opt = tf.train.AdamOptimizer(self.nn_config['lr']).minimize(tf.reduce_mean(joint_losses))
        graph.add_to_collection('opt', opt)
        return opt

    def lookup_table(self, X, mask, graph):
        """
        :param X: shape = (batch_size, words numbers)
        :param mask: used to prevent update of #PAD#
        :return: shape = (batch_size, words numbers, word dim)
        """
        table = tf.placeholder(shape=(2074276, 300), dtype='float32')
        graph.add_to_collection('table', table)
        table = tf.Variable(table, name='table')

        embeddings = tf.nn.embedding_lookup(table, X, partition_strategy='mod', name='lookup_table')
        embeddings = tf.multiply(embeddings,mask)
        graph.add_to_collection('lookup_table', embeddings)
        return embeddings

    def is_word_padding_input(self,X,graph):
        """
        To make the sentence have the same length, we need to pad each sentence with '#PAD#'. To avoid updating of the vector,
        we need a mask to multiply the result of lookup table.
        :param graph: 
        :return: shape = (review number, sentence number, words number)
        """
        ones = tf.ones_like(X, dtype='int32')*2074275
        is_one = tf.equal(X, ones)
        mask = tf.where(is_one, tf.zeros_like(X, dtype='float32'), tf.ones_like(X, dtype='float32'))
        mask = tf.tile(tf.expand_dims(mask, axis=3), multiples=[1,1,1,200])
        return mask

    def classifier(self):
        graph = tf.Graph()
        with graph.as_default():
            R = self.reviews_input(graph=graph)
            words_pad_M = self.is_word_padding_input(R, graph)
            R = self.lookup_table(R, words_pad_M,graph)
            # one hot, 0 represent the setence is padded.
            ispad_M = self.is_sentence_padding_input(graph)
            y_att = self.attribute_labels_input(graph=graph)
            A, o = self.af.attribute_vec(graph)
        sentences_loss = []
        sentences_atr_score = []
        # shape = (batch size, number of sentences, number of attributes)
        sentences_relevance_weight = []
        for k in range(self.nn_config['batch_size']):
            with graph.as_default():
                X = R[k]
                atr_label = y_att[k]
                # ispad.shape = (reviews batch size, number of sentences)
                ispad = ispad_M[k]
                # lstm
                if k > 0:
                    reuse = True
                else:
                    reuse = None
                with tf.variable_scope('sentence_lstm', reuse=reuse):
                    # H.shape = (sentences_num, max_time, cell size)
                    H = self.sentence_lstm(X, reuse=reuse, graph=graph)
                    # get last hidden layer of all sentences
                    # Hl.shape = (sentences number, lstm cell size)
                    Hl = tf.reshape(tf.transpose(H, [1, 0, 2])[-1], shape=(self.nn_config['sentences_num'],
                                                                           self.nn_config['lstm_cell_size']))
                    # calculate relevance score: s(x; a, y)
                    relevance_score = self.Sr(Hl, A, graph)
                # p(x|a,y)
                # shape = (number of attributes, number of sentences)
                sentence_relevance_weight = self.sentence_weight(relevance_score=relevance_score, mask=ispad,
                                                                 graph=graph)
                # shape = (number of sentences, number of attributes), each scalar is p(x|a,y)
                sentence_relevance_weight = tf.transpose(sentence_relevance_weight)
                sentences_relevance_weight.append(sentence_relevance_weight)
            # joint_losses.shape = (number of sentences, attributes number)
            for i in range(self.nn_config['sentences_num']):
                with graph.as_default():
                    # x.shape=(words number, word dim)
                    x = X[i]
                    h = H[i]
                    # shape = (number of attributes,)
                    srw = sentence_relevance_weight[i]
                # attribute function
                with graph.as_default():
                    # shape = (number of attributes, )
                    atr_score_h = self.af.score(A_mat=A, o_mat=o, x=h, graph=graph)
                    atr_score_x = self.af.score(A_mat=A, o_mat=o, x=x, graph=graph)
                    atr_score = tf.add(atr_score_x,atr_score_h)
                    sentences_atr_score.append(atr_score)
                    atr_loss = self.af.loss(atr_score, atr_label, srw, graph)
                    sentences_loss.append(atr_loss)
        with graph.as_default():
            preds = self.af.prediction(sentences_atr_score, sentences_relevance_weight, graph)
            # ispad.shape = (batch size * number of sentences)
            ispad = tf.reshape(ispad_M, shape=(-1,))
            # use ispad to set padded sentences' loss to zero
            loss = tf.reduce_mean(ispad * sentences_loss) + tf.reduce_mean(tf.get_collection('reg'))
            opt = self.optimizer(loss=loss, graph=graph)
            saver = tf.train.Saver()
        return graph, saver

    def train(self):
        graph, saver = self.classifier()
        with graph.as_default():
            # input
            # reviews
            R = graph.get_collection('R')[0]
            # sentence padding
            ispad_M = graph.get_collection('sentences_padding')[0]
            # table
            table = graph.get_collection('table')[0]
            # labels
            y_att = graph.get_collection('y_att')[0]
            # prediction
            pred = graph.get_collection('reviews_atr_pred')[0]
            # train_step
            train_step = graph.get_collection('opt')[0]
            # attribute function
            init = tf.global_variables_initializer()
        table_in = self.dg.table_generator()
        with graph.device('/gpu:0'):
            with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                sess.run(init, feed_dict={table: table_in})
                for i in range(self.nn_config['epoch']):
                    R_data, ispad_M_data, y_att_data = self.dg.data_generator(mode='train', batch_num=i)
                    sess.run(train_step, feed_dict={R: R_data, y_att: y_att_data, ispad_M: ispad_M_data})
                    if i != 0 and i % 1000 == 0:
                        R_data, ispad_M_data, y_att_data = self.dg.data_generator(mode='test')
                        preds = sess.run(pred, feed_dict={R: R_data, y_att: y_att_data, ispad_M: ispad_M_data})
                        mtr = Metrics(preds, y_att_data, self.nn_config)
                        accuracy = mtr.eval()
                        print('accuracy: {}'.format(str(accuracy)))