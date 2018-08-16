import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
from sentiment.functions.attribute_function.attribute_function import AttributeFunction
from sentiment.functions.train.fine_atr_train import FineAtrTrain
from sentiment.functions.attribute_function.metrics import Metrics

import tensorflow as tf


class Classifier:
    def __init__(self, nn_config, data_feeder):
        self.nn_config = nn_config
        self.df = data_feeder
        self.af = AttributeFunction(nn_config)
        self.tra = FineAtrTrain(nn_config, data_feeder)
        self.mt = Metrics(self.nn_config)

    def classifier(self):
        graph = tf.Graph()
        with graph.as_default():
            X_ids = self.af.sentences_input(graph=graph)
            words_pad_M = self.af.is_word_padding_input(X_ids, graph)
            X = self.af.lookup_table(X_ids, words_pad_M, graph)
            # lstm
            with tf.variable_scope('sentence_bilstm'):
                seq_len = self.af.sequence_length(X_ids, graph)
                # H.shape = (batch size, max_time, cell size)
                H = self.af.sentence_bilstm(X, seq_len, graph=graph)
            # Y_att.shape = (batch size, number of attributes+1)
            Y_att = self.af.attribute_labels_input(graph=graph)
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
                o_e = self.af.words_nonattribute_mat2vec(X,o,graph)
                A_e = A_e-o_e
            if not self.nn_config['is_mat']:
                mask = self.af.mask_for_pad_in_score(X_ids,graph)
                score_lstm = self.af.score(A, H,mask, graph)
                # score.shape = (batch size, attributes num, words num)
                score_e = self.af.score(A,X,mask,graph)
                # score.shape = (batch size, attributes num, words num)
                score = tf.add(score_lstm,score_e)
                graph.add_to_collection('score_pre', score)
            else:
                mask = self.af.mask_for_pad_in_score(X_ids, graph)
                score_lstm = self.af.score(A_lstm,H,mask,graph)
                # score.shape = (batch size, attributes num, words num)
                score_e = self.af.score(A_e,X,mask,graph)
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
        classifier = self.classifier()
        self.tra.train(classifier)
