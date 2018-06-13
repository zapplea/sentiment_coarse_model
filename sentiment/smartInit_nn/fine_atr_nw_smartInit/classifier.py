import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
from sentiment.smartInit_nn.smart_init.smart_initiator import SmartInitiator
from sentiment.functions.attribute_function.attribute_function import AttributeFunction
from sentiment.functions.train.smartInit_train import SmartInitTrain
from sentiment.functions.attribute_function.metrics import Metrics

import tensorflow as tf

class Classifier:
    def __init__(self, nn_config, data_generator):
        self.nn_config = nn_config
        self.dg = data_generator
        self.af = AttributeFunction(nn_config)
        self.tra = SmartInitTrain(nn_config, data_generator)
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
            smartInit = SmartInitiator(self.nn_config)
            if not self.nn_config['is_mat']:
                A, o = self.af.attribute_vec(graph)
                A = A - o
            else:
                A, o = self.af.attribute_mat(graph)
                # A.shape = (batch size, words num, attributes number, attribute dim)
                A = self.af.words_attribute_mat2vec(H, A, graph)
                o = self.af.words_nonattribute_mat2vec(H, o, graph)
                A = A - o
            # mask
            mask = self.af.mask_for_pad_in_score(X_ids,graph)
            # score.shape = (batch size, attributes num, max sentence length)
            score = self.af.score(A, H,mask, graph)
            graph.add_to_collection('score_pre', score)
            # score.shape = (batch size, attributes num)
            score = tf.reduce_max(score, axis=2)

            # shape = (batch size, attributes number)
            name_list = smartInit.smart_initiater(graph)
            # shape = (batch size, attributes number)
            name_list_score = smartInit.name_list_score(name_list,graph)
            score = tf.add(name_list_score,score)

            graph.add_to_collection('score', score)
            loss = self.af.sigmoid_loss(score,Y_att,graph)
            pred = self.af.prediction(score, graph)
            accuracy = self.mt.accuracy(Y_att, pred, graph)
        with graph.as_default():
            opt = self.af.optimizer(loss, graph=graph)
            saver = tf.train.Saver()
        return graph, saver

    def train(self):
        classifier = self.classifier
        self.tra.train(classifier)