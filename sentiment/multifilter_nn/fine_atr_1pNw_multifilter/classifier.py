import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
from sentiment.multifilter_nn.multi_filter.multi_filter import MultiFilter
from sentiment.functions.attribute_function.attribute_function import AttributeFunction
from sentiment.functions.train.multifilter_train import MultifilterTrain
from sentiment.functions.attribute_function.metrics import Metrics

import tensorflow as tf
import numpy as np

class Classifier:
    def __init__(self, nn_config, data_generator):
        self.nn_config = nn_config
        self.dg = data_generator
        self.tra = MultifilterTrain(nn_config, data_generator)
        self.af = AttributeFunction(nn_config)
        self.mt = Metrics(self.nn_config)

    def classifier(self):
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
                graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                    graph.get_tensor_by_name('sentence_lstm/rnn/basic_lstm_cell/kernel:0')))
            # Y_att.shape = (batch size, number of attributes+1)
            Y_att = self.af.attribute_labels_input(graph=graph)
            mask = self.af.mask_for_pad_in_score(X_ids, graph)
            keep_prob = tf.placeholder(tf.float32)
            graph.add_to_collection('keep_prob_f', keep_prob)
            mf = MultiFilter(self.nn_config)
            multi_score = []
            for filter_size in self.nn_config['filter_size']:
                filter = mf.filter_generator(X_ids, filter_size)
                # X.shape = (batch size, max sentence length, filter_size*word dim)
                X = mf.look_up(X=X, filter=filter, filter_size=filter_size)
                H = mf.look_up(X=H, filter=filter, filter_size=filter_size)
                # conv_X.shape = (batch size, max sentence length, last dim of conv)
                conv_X = mf.convolution(X=X, filter_size=filter_size, keep_prob=keep_prob, graph=graph)
                conv_H = mf.convolution(X=H, filter_size=filter_size, keep_prob=keep_prob, graph=graph)
                if not self.nn_config['is_mat']:
                    A, o = self.af.attribute_vec(graph)
                    A = A - o
                else:
                    A, o = self.af.attribute_mat(graph)
                    # A.shape = (batch size, words num, attributes number, attribute dim)
                    A_lstm = self.af.words_attribute_mat2vec(conv_H, A, graph)
                    o_lstm = self.af.words_nonattribute_mat2vec(conv_H, o, graph)
                    A_lstm = A_lstm - o_lstm
                    A_e = self.af.words_attribute_mat2vec(conv_X, A, graph)
                    o_e = self.af.words_nonattribute_mat2vec(conv_X, o, graph)
                    A_e = A_e - o_e
                if not self.nn_config['is_mat']:
                    # score.shape = (batch size, attributes num, words num)
                    score_lstm = self.af.score(A, conv_H,mask, graph)
                    # score.shape = (batch size, attributes num, words num)
                    score_e = self.af.score(A,conv_X,mask,graph)
                else:
                    # score.shape = (batch size, attributes num, words num)
                    score_lstm = self.af.score(A_lstm,conv_H,mask,graph)
                    # score.shape = (batch size, attributes num, words num)
                    score_e = self.af.score(A_e,conv_X,mask,graph)
                # score.shape = (batch size, attributes num, words num)
                score = tf.add(score_lstm, score_e)
                # score.shape = (batch size, attributes num, words num,1)
                score = tf.expand_dims(score,axis=3)
                multi_score.append(score)
            graph.add_to_collection('multi_score',multi_score)
            # multi_score.shape = (filter numbers, batch size, attributes number, words num,1)
            # multi_kernel_score = (batch size, attributes number, words num, filter numbers)
            multi_kernel_score = tf.concat(multi_score, axis=3)
            # score.shape = (batch size, attributes number ,words num)
            score = tf.reduce_max(multi_kernel_score, axis=3)
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
        classifier = self.classifier()
        self.tra.train(classifier)