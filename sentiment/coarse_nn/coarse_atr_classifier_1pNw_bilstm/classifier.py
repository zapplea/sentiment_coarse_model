import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
from sentiment.functions.attribute_function.attribute_function import AttributeFunction
from sentiment.functions.attribute_function.metrics import Metrics
from sentiment.coarse_nn.relevance_score.relevance_score import RelScore
from sentiment.functions.train.coarse_atr_train import CoarseTrain

import tensorflow as tf

class Classifier:
    def __init__(self, nn_config, data_generator):
        self.nn_config = nn_config
        self.af = AttributeFunction(nn_config)
        self.mt = Metrics(self.nn_config)
        self.tra = CoarseTrain(self.nn_config,data_generator)

    def classifier(self):
        graph = tf.Graph()
        with graph.as_default():
            relscore = RelScore(self.nn_config)
            # X_ids.shape = (batch size * max review length, words num)
            X_ids = relscore.reviews_input(graph=graph)
            words_pad_M = self.af.is_word_padding_input(X_ids, graph)
            X = self.af.lookup_table(X_ids, words_pad_M, graph)
            # lstm
            with tf.variable_scope('sentence_bilstm'):
                seq_len = self.af.sequence_length(X_ids, graph)
                # H.shape = (batch size, max_time, cell size)
                H = self.af.sentence_bilstm(X, seq_len, graph=graph)
            # Y_att.shape = (batch size * max review length, number of attributes)
            # p(a|D)
            aspect_prob = self.af.attribute_labels_input(graph=graph)

            mask_true_label = self.af.mask_for_true_label(X_ids)

            # Y_att.shape = (batch size * max review length, max review length, attributes num)
            Y_att = relscore.aspect_prob2true_label(aspect_prob,mask_true_label)
            # complement aspect probability
            if self.nn_config['complement']=='1':
                aspect_prob = relscore.complement1_aspect_prob(Y_att, aspect_prob)
            elif self.nn_config['complement'] =='2':
                aspect_prob = relscore.complement2_aspect_prob(Y_att, aspect_prob)

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
                # score.shape = (batch size * max review length, attributes num, words num)
                score_e = self.af.score(A,X,mask,graph)
            else:
                mask = self.af.mask_for_pad_in_score(X_ids, graph)
                score_lstm = self.af.score(A_lstm,H,mask,graph)
                # score.shape = (batch size * max review length, attributes num, words num)
                score_e = self.af.score(A_e,X,mask,graph)
            # score.shape = (batch size * max review length, attributes num, words num)
            score = tf.add(score_lstm, score_e)
            tf.add_to_collection('score_pre',score)
            # score.shape = (batch size, attributes num)
            score = tf.reduce_max(score, axis=2)
            tf.add_to_collection('score',score)
            # eliminate the influce of -inf when calculate relevance probability with softmax
            condition = tf.is_inf(score)
            score = tf.where(condition, tf.zeros_like(score), score)
            # aspect_prob.shape = (batch size * max review length ,attributes num)
            aspect_prob = relscore.expand_aspect_prob(aspect_prob, graph)
            # atr_rel_prob = (batch size * max review length, attributes num)
            atr_rel_prob = relscore.relevance_prob_atr(score, graph)
            # coarse score
            # score = relscore.coarse_atr_score(aspect_prob=aspect_prob, rel_prob=atr_rel_prob, atr_score=score)
            # loss.shape=(batch_size*max review length, attributes num)
            loss = relscore.sigmoid_loss(score, Y_att, atr_rel_prob, aspect_prob, graph)
            #loss = tf.multiply(atr_rel_prob,tf.multiply(aspect_prob,loss))
            tf.add_to_collection('coarse_atr_loss',loss)
            pred = self.af.prediction(score, graph)
            accuracy = self.mt.accuracy(Y_att, pred, graph)
        with graph.as_default():
            opt = self.af.optimizer(loss, graph=graph)
            saver = tf.train.Saver()
        return graph, saver

    def train(self):
        self.tra.train(self.classifier)
