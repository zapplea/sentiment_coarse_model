import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.functions.sentiment_function.sentiment_function import SentiFunction
from sentiment.functions.train.sentinn_train import SentiTrain

import tensorflow as tf

class Classifier:
    def __init__(self, nn_config,data_generator):
        self.nn_config = nn_config
        self.dg = data_generator
        self.sf= SentiFunction(self.nn_config)
        self.tra = SentiTrain(self.nn_config, self.dg)

    def classifier(self):
        graph = tf.Graph()
        with graph.as_default():
            X_ids = self.sf.sentences_input(graph=graph)
            words_pad_M = self.sf.is_word_padding_input(X_ids, graph)
            table = self.sf.wordEbmedding_table_input(graph)
            X = self.sf.lookup_table(X_ids,words_pad_M,table,graph)
            # lstm
            with tf.variable_scope('sentence_bilstm'):
                seq_len = self.sf.sequence_length(X_ids, graph)
                # H.shape = (batch size, max_time, cell size)
                H = self.sf.sentence_bilstm(X,seq_len, graph=graph)
                #
            # path dependency
            # PD_ids.shape = ()
            # PD_ids.shape= (batch size, words number, words number, max dp length)
            PD_ids = self.sf.path_dependency_table_input(graph)
            # path_dependence_pad_M.shape= (batch size, words number, words number, max dp length)
            path_dependence_pad_M = self.sf.is_dpword_padding_input(PD_ids,graph)
            # PD.shape = (batch size, words num, words num, max path length, word dim)
            # PD.shape =
            PD = self.sf.lookup_table(PD_ids,
                                      path_dependence_pad_M,
                                      table,
                                      graph)
            with tf.variable_scope('dependency_path_bilstm'):
                seq_len = self.sf.path_sequence_length(PD_ids)
                # pd_H.shape=(batch size, words num, words num, lstm cell size)
                pd_H = self.sf.path_dependency_bilstm(PD,seq_len,graph)
            # TODO:eliminate the influence of padded dependency path; actually, the 0 will not influence.
            Y_att = self.sf.attribute_labels_input(graph=graph)
            Y_senti = self.sf.sentiment_labels_input(graph=graph)
            if not self.nn_config['is_mat']:
                A = self.sf.attribute_vec(graph)
            else:
                A = self.sf.attribute_mat(graph)
                # A.shape = shape = (batch size, number of words, number of attributes + 1, attribute dim(=lstm cell dim))
                A = self.sf.words_attribute_mat2vec(H=H, A_mat=A, graph=graph)
            # sentiment expression prototypes matrix
            # shape = (3*numbers of normal sentiment prototype + attributes_numbers*attribute specific sentiment prototypes)
            W = self.sf.sentiment_matrix(graph)
            # sentiment extractors for all (yi,ai)
            # extors_mat.shape = (3*attributes number+3, sentiment prototypes number, sentiment dim)
            extors_mat = self.sf.senti_extors_mat(graph)
            # extors_mask_mat.shape = (3*attributes number+3, sentiment prototypes number)
            extors_mask_mat = self.sf.extors_mask(extors=extors_mat,graph=graph)
            beta = self.sf.beta(graph)
            # extract sentiment expression corresponding to sentiment and attribute from W for all attributes
            # W.shape=(number of attributes*3+3, size of original W); shape of original W =(3*normal sentiment prototypes + attribute number * attribute sentiment prototypes, sentiment dim)
            W = tf.multiply(extors_mat, W)
            # shape = (batch size,number of words, 3+3*attributes number, number of sentiment prototypes)
            attention = self.sf.sentiment_attention(H, W, extors_mask_mat, graph)
            # attended_W.shape = (batch size,number of words, 3+3*attributes number, sentiment dim)
            attended_W = self.sf.attended_sentiment(W, attention, graph)
            # shape = (batch size,number of words, 3+3*attributes number)
            item1 = self.sf.item1(attended_W,H,graph)
            # A_dist.shape = (batch size, number of attributes+1, wrods number)
            A_dist = self.sf.attribute_distribution(A=A, H=H, graph=graph)
            # mask for score to eliminate the influence of padding word
            mask = self.sf.mask_for_pad_in_score(X_ids, graph)

            # TODO: dependency path
            # A_Vi.shape = (batch size, number of attributes+1, number of words, relative position dim)
            dp_A_Vi = self.sf.dp_Vi(A_dist=A_dist,PD=pd_H,graph=graph)
            # item2.shape=(batch size, number of attributes+1, number of words)
            item2 = tf.reduce_sum(tf.multiply(dp_A_Vi, beta), axis=3)
            # dp_socre.shape = (batch size, 3*number of attributes+3, words num)
            dp_score = self.sf.score(item1, item2,mask, graph)
            # TODO: relative distance
            V = self.sf.relative_pos_matrix(graph)
            # relative position ids
            # shape = (number of words, number of words)
            rp_ids = self.sf.relative_pos_ids(graph)
            # A_Vi.shape = (batch size, number of attributes+1, number of words, relative position dim)
            rd_A_Vi = self.sf.rd_Vi(A_dist=A_dist, V=V, rp_ids=rp_ids, graph=graph)
            # item2.shape=(batch size, number of attributes+1, number of words)
            item2 = tf.reduce_sum(tf.multiply(rd_A_Vi, beta), axis=3)
            # rd_socre.shape = (batch size, 3*number of attributes+3, words num)
            rd_score = self.sf.score(item1, item2, mask, graph)
            # TODO: combine scores of relative distance and dependency path
            # score.shape = (batch size, 3*number of attributes+3, words num)
            score = tf.add(rd_score,dp_score)

            # score.shape = (batch size, 3*number of attributes+3)
            score = tf.reduce_max(score, axis=2)
            # in coarse model, when the whole sentence is padded, there will be -inf, so need to convert them to 0
            condition = tf.is_inf(score)
            score = tf.where(condition, tf.zeros_like(score), score)
            graph.add_to_collection('senti_score', score)
            # # max_false_score.shape = (batch size, attributes number, 3)
            # max_false_score = self.sf.max_false_senti_score(Y_senti, score, graph)
            # #
            # senti_loss = self.sf.loss(Y_senti, score, max_false_score, graph)
            # opt = self.sf.optimizer(senti_loss,graph)
            # senti_pred = self.sf.prediction(score=score, Y_atr=Y_att, graph=graph)
            # mask the situation when attribute doesn't appear
            mask = tf.tile(tf.expand_dims(Y_att, axis=2), multiples=[1, 1, 3])
            # score.shape = (batch size, number of attributes+1,3)
            score = tf.multiply(tf.reshape(score, shape=(-1, self.nn_config['attributes_num'] + 1, 3)), mask)
            # softmax loss
            # TODO: check reg
            senti_loss = self.sf.softmax_loss(labels=Y_senti, logits=score, graph=graph)
            opt = self.sf.optimizer(senti_loss, graph)
            # TODO: in coarse, should mask the prediction of padded sentences.
            senti_pred = self.sf.prediction(score=score, Y_atr=Y_att, graph=graph)

            saver = tf.train.Saver()
        return graph, saver

    def train(self):
        self.tra.train(self.classifier)