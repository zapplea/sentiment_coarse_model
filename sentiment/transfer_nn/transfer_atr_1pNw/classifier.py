import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
from sentiment.functions.attribute_function.attribute_function import AttributeFunction
from sentiment.functions.attribute_function.metrics import Metrics
from sentiment.coarse_nn.coarse_atr_classifier_1pNw.classifier import Classifier as coarse_Classifier
from sentiment.transfer_nn.transfer.transfer import Transfer
from sentiment.functions.train.trans_atr_train import TransferTrain

import tensorflow as tf

class Classifier:
    def __init__(self, coarse_nn_config, fine_nn_config, coarse_data_generator, fine_data_generator):
        self.coarse_nn_config = coarse_nn_config
        self.coarse_dg = coarse_data_generator
        self.coarse_cl = coarse_Classifier(coarse_nn_config,coarse_data_generator)

        self.fine_nn_config = fine_nn_config
        self.fine_dg = fine_data_generator
        self.af = AttributeFunction(self.fine_nn_config)
        self.mt = Metrics(self.fine_nn_config)

        self.trans=Transfer(self.fine_nn_config,coarse_nn_config,self.coarse_dg)
        self.tra=TransferTrain(fine_nn_config,self.fine_dg)

    def transfer(self):
        initializer_A_data,initializer_O_data = self.trans.transfer(self.coarse_cl,self.fine_dg)
        return initializer_A_data,initializer_O_data


    def fine_classifier(self):
        graph = tf.Graph()
        with graph.as_default():
            X_ids = self.af.sentences_input(graph=graph)
            words_pad_M = self.af.is_word_padding_input(X_ids, graph)
            X = self.af.lookup_table(X_ids, words_pad_M, graph)
            # lstm
            with tf.variable_scope('sentence_lstm'):
                seq_len = self.af.sequence_length(X_ids, graph)
                # H.shape = (batch size, max_time, cell size)
                H = self.af.sentence_lstm(X, seq_len, graph=graph)
                graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.fine_nn_config['reg_rate'])(
                    graph.get_tensor_by_name('sentence_lstm/rnn/basic_lstm_cell/kernel:0')))
            # Y_att.shape = (batch size, number of attributes+1)
            Y_att = self.af.attribute_labels_input(graph=graph)
            if not self.fine_nn_config['is_mat']:
                A, o = self.trans.attribute_vec(graph=graph)
                A = A - o
            else:
                A, o = self.trans.attribute_mat(graph=graph)
                # A.shape = (batch size, words num, attributes number, attribute dim)
                A_lstm = self.af.words_attribute_mat2vec(H, A, graph)
                o_lstm = self.af.words_nonattribute_mat2vec(H, o, graph)
                A_lstm = A_lstm - o_lstm
                A_e = self.af.words_attribute_mat2vec(X, A, graph)
                o_e = self.af.words_nonattribute_mat2vec(X, o, graph)
                A_e = A_e - o_e
            if not self.fine_nn_config['is_mat']:
                mask = self.af.mask_for_pad_in_score(X_ids, graph)
                score_lstm = self.af.score(A, H, mask, graph)
                # score.shape = (batch size, attributes num, words num)
                score_e = self.af.score(A, X, mask, graph)
                # score.shape = (batch size, attributes num, words num)
                score = tf.add(score_lstm, score_e)
                graph.add_to_collection('score_pre', score)
            else:
                mask = self.af.mask_for_pad_in_score(X_ids, graph)
                score_lstm = self.af.score(A_lstm, H, mask, graph)
                # score.shape = (batch size, attributes num, words num)
                score_e = self.af.score(A_e, X, mask, graph)
                # score.shape = (batch size, attributes num, words num)
                score = tf.add(score_lstm, score_e)
                graph.add_to_collection('score_pre', score)
            # score.shape = (batch size, attributes num)
            score = tf.reduce_max(score, axis=2)
            graph.add_to_collection('score', score)
            loss = self.af.sigmoid_loss(score, Y_att, graph)
            pred = self.af.prediction(score, graph)
            accuracy = self.mt.accuracy(Y_att, pred, graph)
        with graph.as_default():
            opt = self.af.optimizer(loss, graph=graph)
            saver = tf.train.Saver()
        return graph, saver

    def train(self):
        """
        train fine grained model with 
        :return: 
        """
        cl = self.fine_classifier
        initializer_A_data , initializer_O_data = self.transfer()
        self.tra.train(cl,initializer_A_data=initializer_A_data,initializer_O_data=initializer_O_data)