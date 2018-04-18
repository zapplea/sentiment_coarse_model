import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
from sentiment.smartInit_nn.smart_init.smart_initiator import SmartInitiator

import tensorflow as tf
import numpy as np
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer


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

    def attribute_mat(self, smartInit, graph):
        """

        :param graph: 
        :return: shape = (attributes number+1, attribute mat size, attribute dim)
        """
        A_mat = tf.get_variable(name='A_mat', initializer=smartInit)
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(A_mat))
        graph.add_to_collection('A_mat', A_mat)
        o_mat = tf.get_variable(name='other_vec',
                                initializer=tf.random_uniform(shape=(1,
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

    def score(self, A, X,mask, graph):
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
            # mask.shape = (batch size, attributes number, words num)
            mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, self.nn_config['attributes_num'],1])
            score = tf.add(score, mask)
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
        return score

    def prediction(self, score, graph):
        condition = tf.greater(score, tf.ones_like(score, dtype='float32') * self.nn_config['atr_pred_threshold'])
        pred = tf.where(condition, tf.ones_like(score, dtype='float32'), tf.zeros_like(score, dtype='float32'))
        graph.add_to_collection('atr_pred', pred)
        return pred

    def TN(self, Y_att, pred, graph):
        """

        :param Y_att: shape = (batch size, attributes number)
        :param pred: shape = (batch size, attributes number)
        :param graph: 
        :return: 
        """


        TP = tf.cast(tf.count_nonzero(pred * Y_att), tf.float32)

        TN = tf.cast(tf.count_nonzero((pred - 1) * (Y_att - 1)), tf.float32)
        FP = tf.cast(tf.count_nonzero(pred * (Y_att - 1)), tf.float32)
        graph.add_to_collection('TP', TP)
        graph.add_to_collection('FP', FP)

        FN = tf.cast(tf.count_nonzero((pred - 1) * Y_att), tf.float32)
        graph.add_to_collection('FN', FN)

        return f1


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
        # create a mask for non-attribute in which Sa is 0 and need to keep max false attribute
        Y_temp = tf.reduce_sum(Y_att, axis=1)
        condition = tf.equal(tf.reduce_sum(Y_att, axis=1),
                             tf.zeros_like(Y_temp, dtype='float32'))
        item1 = tf.tile(tf.expand_dims(
            tf.multiply(tf.ones_like(Y_temp, dtype='float32'), tf.divide(1, self.nn_config['attributes_num'])), axis=1),
            multiples=[1, self.nn_config['attributes_num']])
        nonatr_mask = tf.where(condition, item1, Y_att)
        #
        theta = tf.constant(self.nn_config['attribute_loss_theta'], dtype='float32')
        # loss.shape = (batch size, attributes num)
        loss = tf.multiply(tf.add(tf.subtract(theta, tf.multiply(Y_att, score)), max_fscore), nonatr_mask)
        zero_loss = tf.zeros_like(loss, dtype='float32')

        loss = tf.expand_dims(loss, axis=2)
        zero_loss = tf.expand_dims(zero_loss, axis=2)
        # loss.shape = (batch size, attributes num)
        loss = tf.reduce_max(tf.concat([loss, zero_loss], axis=2), axis=2)
        # The following is obsolete design of loss function with regularization.
        # loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1)) + tf.multiply(1 / self.nn_config['batch_size'],
        #                                                                  tf.reduce_sum(graph.get_collection('reg')))
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1) + tf.reduce_sum(graph.get_collection('reg')))
        graph.add_to_collection('atr_loss', loss)
        tf.summary.scalar('loss', loss)

        return loss


class Classifier:
    def __init__(self, nn_config, data_generator):
        self.nn_config = nn_config
        self.dg = data_generator
        self.af = AttributeFunction(nn_config)

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
        paddings = tf.ones_like(X, dtype='int32')*self.nn_config['padding_word_index']
        condition = tf.equal(paddings, X)
        seq_len = tf.reduce_sum(tf.where(condition, tf.zeros_like(X, dtype='int32'), tf.ones_like(X, dtype='int32')),
                                axis=1, name='seq_len')
        return seq_len

    def mask_for_pad_in_score(self,X,graph):
        """
        This mask is used in score, to eliminate the influence of pad words when reduce_max. This this mask need to add to the score.
        Since 0*inf = nan
        :param X: the value is word id. shape=(batch size, max words num)
        :param graph: 
        :return: 
        """
        paddings = tf.ones_like(X, dtype='int32') * self.nn_config['padding_word_index']
        condition = tf.equal(paddings, X)
        mask = tf.where(condition, tf.ones_like(X, dtype='float32')*tf.convert_to_tensor(-np.inf), tf.zeros_like(X, dtype='float32'))
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
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size'])
        # outputs.shape = (batch size, max_time, cell size)
        outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=X, time_major=False,sequence_length=seq_len,dtype='float32')
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
            X_ids = self.sentences_input(graph=graph)
            words_pad_M = self.is_word_padding_input(X_ids, graph)
            X = self.lookup_table(X_ids, words_pad_M, graph)
            # lstm
            with tf.variable_scope('sentence_lstm'):
                seq_len = self.sequence_length(X_ids, graph)
                # H.shape = (batch size, max_time, cell size)
                H = self.sentence_lstm(X, seq_len, graph=graph)
                graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                    graph.get_tensor_by_name('sentence_lstm/rnn/basic_lstm_cell/kernel:0')))
            # Y_att.shape = (batch size, number of attributes+1)
            Y_att = self.attribute_labels_input(graph=graph)
            smartInit = SmartInitiator(self.nn_config)

            if not self.nn_config['is_mat']:
                A, o = self.af.attribute_vec(graph)
                A = A - o
            else:
                A, o = self.af.attribute_mat(smartInit.smart_initiater(graph), graph)
                # A.shape = (batch size, words num, attributes number, attribute dim)
                A_lstm = self.af.words_attribute_mat2vec(H, A, graph)
                o_lstm = self.af.words_nonattribute_mat2vec(H, o, graph)
                A_lstm = A_lstm - o_lstm
                A_e = self.af.words_attribute_mat2vec(X,A,graph)
                o_e = self.af.words_nonattribute_mat2vec(X,o,graph)
                A_e = A_e-o_e
            if not self.nn_config['is_mat']:
                mask = self.mask_for_pad_in_score(X_ids,graph)
                score_lstm = self.af.score(A, H,mask, graph)
                # score.shape = (batch size, attributes num, words num)
                score_e = self.af.score(A,X,mask,graph)
                # score.shape = (batch size, attributes num, words num)
                score = tf.add(score_lstm,score_e)
                graph.add_to_collection('score_pre', score)
                # score.shape = (batch size, attributes num)
                score = tf.reduce_max(score, axis=2)
            else:
                mask = self.mask_for_pad_in_score(X_ids, graph)
                score_lstm = self.af.score(A_lstm,H,mask,graph)
                # score.shape = (batch size, attributes num, words num)
                score_e = self.af.score(A_e,X,mask,graph)
                # score.shape = (batch size, attributes num, words num)
                score = tf.add(score_lstm, score_e)
                graph.add_to_collection('score_pre', score)
                # score.shape = (batch size, attributes num)
                score = tf.reduce_max(score, axis=2)
            graph.add_to_collection('score', score)
            max_fscore = self.af.max_false_score(score, Y_att, graph)
            loss = self.af.loss(score, max_fscore, Y_att, graph)
            pred = self.af.prediction(score, graph)
            # micro = self.af.micro_average(Y_att, pred, graph)
            macro = self.af.macro_average(Y_att, pred, graph)
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
            train_step = graph.get_collection('opt')[0]
            #
            table = graph.get_collection('table')[0]
            #
            loss = graph.get_collection('atr_loss')[0]

            pred = graph.get_collection('atr_pred')[0]

            smartInit = graph.get_collection('smartInit')[0]
            score = graph.get_collection('score')[0]
            score_pre = graph.get_collection('score_pre')[0]
            max_false_score = graph.get_collection('max_false_score')[0]
            TP = graph.get_collection('TP')[0]
            FN = graph.get_collection('FN')[0]
            FP = graph.get_collection('FP')[0]
            # attribute function
            init = tf.global_variables_initializer()
        table_data = self.dg.table
        smartInit_data = self.dg.smart_init_embedding
        print(self.dg.aspect_dic)

        with graph.device('/gpu:1'):
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(graph=graph, config=config) as sess:
                sess.run(init, feed_dict={smartInit: smartInit_data,table: table_data})

                batch_num = int(self.dg.train_data_size / self.nn_config['batch_size'])
                print('Train set size: ', self.dg.train_data_size, 'Test set size:', self.dg.test_data_size)
                for i in range(self.nn_config['epoch']):
                    loss_vec = []
                    pred_vec = []
                    score_vec = []
                    score_pre_vec = []
                    max_false_score_vec = []
                    Y_att_vec  = []
                    TP_vec = []
                    FP_vec = []
                    FN_vec = []
                    for j in range(batch_num):
                        sentences, Y_att_data = self.dg.train_data_generator(j,i)
                        _, train_loss,TP_data, FP_data, FN_data, pred_data, score_data, max_false_score_data, score_pre_data \
                            = sess.run(
                            [train_step, loss, TP,FP,FN,pred, score, max_false_score, score_pre  ],
                            feed_dict={X: sentences, Y_att: Y_att_data})

                        ###Show training message
                        loss_vec.append(train_loss)
                        TP_vec.append(TP_data)
                        FP_vec.append(FP_data)
                        FN_vec.append(FN_data)
                        for n in range(self.nn_config['batch_size']):
                            pred_vec.append(pred_data[n])
                            score_vec.append(score_data[n])
                            score_pre_vec.append(score_pre_data[n])
                            max_false_score_vec.append(max_false_score_data[n])
                            Y_att_vec.append(Y_att_data[n])
                    if i % 20 == 0:
                        check_num = 1
                        print('Epoch:', i, '\nTraining loss:%.10f' % np.mean(loss_vec))

                        _precision = self.precision(TP_vec,FP_vec,'macro')
                        _recall = self.recall(TP_vec,FN_vec,'macro')
                        _f1_score = self.f1_score(_precision,_recall,'macro')
                        print('F1 score for each class:',_f1_score,'\nPrecison for each class:',_precision,'\nRecall for each class:',_recall)
                        print('Macro F1 sorce:',np.mean(_f1_score) ,' Macro precision:', np.mean(_precision),' Macro recall:', np.mean(_recall) )

                        _precision = self.precision(TP_vec, FP_vec, 'micro')
                        _recall = self.recall(TP_vec, FN_vec, 'micro')
                        _f1_score = self.f1_score(_precision, _recall, 'micro')
                        print('Micro F1 sorce:', _f1_score, ' Micro precision:', np.mean(_precision), ' Micro recall:', np.mean(_recall))

                        # # np.random.seed(1)
                        # random_display = np.random.randint(0, 1700, check_num)
                        # pred_check = [[list(self.dg.aspect_dic.keys())[c] for c, rr in enumerate(pred_vec[r]) if rr] for
                        #               r in random_display]
                        # sentences_check = [
                        #     [list(self.dg.dictionary.keys())[word] for word in self.dg.train_sentence_ground_truth[r] if word] for r
                        #     in random_display]
                        # Y_att_check = [[list(self.dg.aspect_dic.keys())[c] for c, rr in
                        #                 enumerate(self.dg.train_attribute_ground_truth[r]) if rr] for r in
                        #                random_display]
                        # score_check = [score_vec[r] for r in random_display]
                        # score_pre_check = [score_pre_vec[r] for r in random_display]
                        # max_false_score_check = [max_false_score_vec[r] for r in random_display]
                        # for n in range(check_num):
                        #     print("sentence id: ", random_display[n], "\nsentence:\n", sentences_check[n], "\npred:\n",
                        #           pred_check[n],
                        #           "\nY_att:\n", Y_att_check[n]
                        #           , "\nscore:\n", score_check[n], "\nmax_false_score:\n", max_false_score_check[n])
                        #     for nn in range(len(score_pre_check[n])):
                        #         if list(self.dg.aspect_dic.keys())[nn] in Y_att_check[n]:
                        #             print(list(self.dg.aspect_dic.keys())[nn] + " score:", score_pre_check[n][nn])

                    if i % 200 == 0 and i != 0:
                        print('Test.....')
                        sentences, Y_att_data = self.dg.test_data_generator()
                        valid_size = Y_att_data.shape[0]
                        loss_vec = []
                        pred_vec = []
                        score_vec = []
                        score_pre_vec = []
                        max_false_score_vec = []
                        Y_att_vec = []
                        TP_vec = []
                        FP_vec = []
                        FN_vec = []
                        batch_size = self.nn_config['batch_size']
                        for i in range(valid_size // batch_size):
                            test_loss,  pred_data, score_data, max_false_score_data, score_pre_data,TP_data, FP_data, FN_data  = sess.run([loss, pred, score, max_false_score, score_pre,TP,FP,FN],
                                                                                                                                feed_dict={X: sentences[i * batch_size:i * batch_size + batch_size],
                                                                                                                               Y_att: Y_att_data[i * batch_size:i * batch_size + batch_size]
                                                                                                                               })
                            ###Show test message
                            TP_vec.append(TP_data)
                            FP_vec.append(FP_data)
                            FN_vec.append(FN_data)
                            loss_vec.append(test_loss)
                            for n in range(self.nn_config['batch_size']):
                                pred_vec.append(pred_data[n])
                                score_vec.append(score_data[n])
                                score_pre_vec.append(score_pre_data[n])
                                max_false_score_vec.append(max_false_score_data[n])
                        print('\nTest loss:%.10f' % np.mean(loss_vec))

                        _precision = self.precision(TP_vec, FP_vec, 'macro')
                        _recall = self.recall(TP_vec, FN_vec, 'macro')
                        _f1_score = self.f1_score(_precision, _recall, 'macro')
                        print('F1 score for each class:', _f1_score, '\nPrecison for each class:', _precision,
                              '\nRecall for each class:', _recall)
                        print('Macro F1 sorce:', np.mean(_f1_score), ' Macro precision:', np.mean(_precision),
                              ' Macro recall:', np.mean(_recall))

                        _precision = self.precision(TP_vec, FP_vec, 'micro')
                        _recall = self.recall(TP_vec, FN_vec, 'micro')
                        _f1_score = self.f1_score(_precision, _recall, 'micro')
                        print('Micro F1 sorce:', _f1_score, ' Micro precision:', np.mean(_precision), ' Micro recall:',np.mean(_recall))
                        # # np.random.seed(1)
                        # random_display = np.random.randint(0, 570, check_num)
                        # pred_check = [[list(self.dg.aspect_dic.keys())[c] for c, rr in enumerate(pred_vec[r]) if rr] for
                        #               r in random_display]
                        # sentences_check = [
                        #     [list(self.dg.dictionary.keys())[word] for word in self.dg.test_sentence_ground_truth[r] if
                        #      word] for r
                        #     in random_display]
                        # Y_att_check = [[list(self.dg.aspect_dic.keys())[c] for c, rr in
                        #                 enumerate(self.dg.test_attribute_ground_truth[r]) if rr] for r in
                        #                random_display]
                        # score_check = [score_vec[r] for r in random_display]
                        # score_pre_check = [score_pre_vec[r] for r in random_display]
                        # max_false_score_check = [max_false_score_vec[r] for r in random_display]
                        # for n in range(check_num):
                        #     print("sentence id: ", random_display[n], "\nsentence:\n", sentences_check[n], "\npred:\n",
                        #           pred_check[n],
                        #           "\nY_att:\n", Y_att_check[n]
                        #           , "\nscore:\n", score_check[n], "\nmax_false_score:\n", max_false_score_check[n])
                        #     for nn in range(len(score_pre_check[n])):
                        #         if list(self.dg.aspect_dic.keys())[nn] in Y_att_check[n]:
                        #             print(list(self.dg.aspect_dic.keys())[nn] + " score:", score_pre_check[n][nn])

    def precision(self,TP,FP,flag):
        assert flag=='macro' or flag=='micro','Please enter right flag...'
        if flag == 'macro':
            tmp = np.nonzero((np.sum(TP,axis=0) + np.sum(FP,axis=0) == 0))
            res = np.sum(TP,axis=0,dtype='float32') / ( np.sum(TP,axis=0,dtype='float32') + np.sum(FP,axis=0,dtype='float32') )
            res[tmp] = 1
            return res
        else:
            return np.sum(TP) / ( np.sum(TP) + np.sum(FP) )

    def recall(self,TP,FN,flag):
        assert flag=='macro' or flag=='micro','Please enter right flag...'
        if flag == 'macro':
            tmp = np.nonzero((np.sum(TP, axis=0) + np.sum(FN, axis=0) == 0))
            res = np.sum(TP, axis=0 ,dtype='float32') / (np.sum(TP, axis=0,dtype='float32') + np.sum(FN, axis=0,dtype='float32'))
            res[tmp] = 1
            return res
        else:
            return np.sum(TP) / ( np.sum(TP) + np.sum(FN) )


    def f1_score(self,precision,recall,flag):
        assert flag=='macro' or flag=='micro','Please enter right flag...'
        if flag == 'macro':
            tmp = np.nonzero((precision + recall) == 0)
            res = 2 * precision * recall / ( precision + recall )
            res[tmp] = 0
            return res
        else:
            return 2 * precision * recall / ( precision + recall )


