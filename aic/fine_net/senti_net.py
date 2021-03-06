import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import tensorflow as tf

from aic.functions.fine_functions import SentimentFunction
from aic.functions.comm_functions import FineCommFunction
from aic.functions.fine_functions import AttributeFunction
from aic.functions.multiGPU_builders import SentiNetBuilder

class SentimentNet:
    def __init__(self,config,graph,table):
        self.nn_config = config
        self.sf = SentimentFunction(self.nn_config)
        self.comm = FineCommFunction(self.nn_config)
        self.af = AttributeFunction(self.nn_config)
        self.graph = graph
        self.table = table
        self.reg = {'attr_reg':[],'senti_reg':[]}
        self.classifier()

    def classifier(self):
        X_ids = self.comm.sentences_input(self.graph)
        # shape = (batch size, number of attributes, 3)
        Y_att = self.comm.attribute_labels_input(self.graph)
        # #################### #
        # attribute extraction #
        # #################### #
        words_pad_M = self.comm.is_word_padding_input(X_ids,self.graph)
        # TODO: lookup table
        X = self.comm.lookup_table(X_ids, words_pad_M,self.table,self.graph)
        # lstm
        with tf.variable_scope('sentence_bilstm',reuse=tf.AUTO_REUSE):
            seq_len = self.comm.sequence_length(X_ids,self.graph)
            # H.shape = (batch size, max_time, cell size)
            attr_H = self.comm.sentence_bilstm('attr_reg',X, seq_len, self.reg,self.graph, scope_name='sentiment/sentence_bilstm')
        A, o = self.af.attribute_mat(self.reg,self.graph)
        # A.shape = (batch size, words num, attributes number, attribute dim)
        A_lstm = self.af.words_attribute_mat2vec(attr_H, A, self.graph)
        o_lstm = self.af.words_nonattribute_mat2vec(attr_H, o, self.graph)
        A_lstm = A_lstm - o_lstm
        A_e = self.af.words_attribute_mat2vec(X, A, self.graph)
        o_e = self.af.words_nonattribute_mat2vec(X, o, self.graph)
        A_e = A_e - o_e

        mask = self.comm.mask_for_pad_in_score(X_ids, self.graph)
        score_lstm = self.af.score(A_lstm, attr_H, mask, self.graph)
        # score.shape = (batch size, attributes num, words num)
        score_e = self.af.score(A_e, X, mask, self.graph)
        # score.shape = (batch size, attributes num, words num)
        score = tf.add(score_lstm, score_e)

        # score.shape = (batch size, attributes num)
        score = tf.reduce_max(score, axis=2)
        reg_list = []
        for reg in self.reg['attr_reg']:
            reg_list.append(reg)
        attr_loss = self.af.sigmoid_loss('attr_loss',score, Y_att, reg_list,self.graph)
        attr_pred_labels = self.af.prediction('attr_pred_labels',score, self.graph)


        # #################### #
        # sentiment extraction #
        # #################### #
        # sentiment lstm
        with tf.variable_scope('senti_sentence_bilstm',reuse=tf.AUTO_REUSE):
            # H.shape = (batch size, max_time, cell size)
            senti_H = self.comm.sentence_bilstm('senti_reg',X, seq_len, self.reg, self.graph, scope_name='sentiment/sentence_bilstm')
        # Y_senti.shape = [batch_size, number of attributes + 1, 3]
        Y_senti = self.comm.sentiment_labels_input(self.graph)
        # sentiment expression prototypes matrix
        # shape = (3*numbers of normal sentiment prototype + attributes_numbers*attribute specific sentiment prototypes)
        W = self.sf.sentiment_matrix(self.reg, self.graph)
        # sentiment extractors for all (yi,ai)
        # extors_mat.shape = (3*attributes number+3, sentiment prototypes number, sentiment dim)
        extors_mat = self.sf.senti_extors_mat(self.graph)
        # extors_mask_mat.shape = (3*attributes number+3, sentiment prototypes number)
        extors_mask_mat = self.sf.extors_mask(extors=extors_mat,graph=self.graph)
        # relative position matrix
        V = self.sf.relative_pos_matrix(self.reg,self.graph)
        # relative position ids
        # shape = (number of words, number of words)
        rp_ids = self.sf.relative_pos_ids(self.graph)
        beta = self.sf.beta(self.reg, self.graph)
        # extract sentiment expression corresponding to sentiment and attribute from W for all attributes
        # W.shape=(number of attributes*3+3, size of original W); shape of original W =(3*normal sentiment prototypes + attribute number * attribute sentiment prototypes, sentiment dim)
        W = tf.multiply(extors_mat, W)
        # shape = (batch size,number of words, 3+3*attributes number, number of sentiment prototypes)
        attention = self.sf.sentiment_attention(senti_H, W, extors_mask_mat, self.graph)
        # attended_W.shape = (batch size,number of words, 3+3*attributes number, sentiment dim)
        attended_W = self.sf.attended_sentiment(W, attention, self.graph)
        # shape = (batch size,number of words, 3+3*attributes number)
        item1 = self.sf.item1(attended_W, senti_H, self.graph)
        # A_dist.shape = (batch size, number of attributes+1, wrods number)
        if self.nn_config['is_mat']:
            A = tf.concat([A, o], axis=0)
            # A.shape = shape = (batch size, number of words, number of attributes + 1, attribute dim(=lstm cell dim))
            A = self.sf.words_attribute_mat2vec(H=attr_H, A_mat=A,graph=self.graph)

        A_dist = self.sf.attribute_distribution(A=A, H=attr_H, graph=self.graph)
        # A_Vi.shape = (batch size, number of attributes+1, number of words, relative position dim)
        A_Vi = self.sf.rd_Vi(A_dist=A_dist, V=V, rp_ids=rp_ids, graph=self.graph)
        # item2.shape=(batch size, number of attributes+1, number of words)
        item2 = tf.reduce_sum(tf.multiply(A_Vi, beta), axis=3)
        # mask for score to eliminate the influence of padded word
        mask = self.sf.mask_for_pad_in_score(X_ids, self.graph)
        # senti_socre.shape = (batch size, 3*number of attributes+3, words num)
        score = self.sf.score(item1, item2, mask, self.graph)
        # score.shape = (batch size, 3*number of attributes+3)
        score = tf.reduce_max(score, axis=2)

        # pure senti loss
        # mask the situation when attribute doesn't appear
        Y_att = self.sf.expand_attr_labels(Y_att, self.graph)
        # the mask is from attribute true labels
        mask = tf.tile(tf.expand_dims(Y_att, axis=2), multiples=[1, 1, 3])
        # score.shape = (batch size, number of attributes+1,3)
        pure_score = tf.multiply(tf.reshape(score, shape=(-1, self.nn_config['attributes_num'] + 1, 3)), mask)
        # softmax loss
        # TODO: check reg
        reg_list=[]
        for reg in self.reg['senti_reg']:
            reg_list.append(reg)
        senti_loss = self.sf.softmax_loss(name='senti_loss',labels=Y_senti, logits=pure_score,reg_list=reg_list, graph=self.graph)
        senti_pred_labels = self.sf.prediction(name='senti_pred_labels',score=pure_score, Y_atr=Y_att,graph=self.graph)

        # joint loss
        # mask the situation when attribute doesn't appear
        attr_pred_labels = self.sf.expand_attr_labels(attr_pred_labels, self.graph)
        # the mask is from attribute prediction labels
        mask = tf.tile(tf.expand_dims(attr_pred_labels, axis=2), multiples=[1, 1, 3])
        # score.shape = (batch size, number of attributes+1,3)
        joint_score = tf.multiply(tf.reshape(score, shape=(-1, self.nn_config['attributes_num'] + 1, 3)), mask)
        # softmax loss
        # TODO: loss of joint should be???
        senti_loss_of_joint = self.sf.softmax_loss(name='senti_loss_of_joint',labels=Y_senti, logits=joint_score,reg_list=reg_list,graph=self.graph)
        joint_loss = senti_loss_of_joint + attr_loss
        self.graph.add_to_collection('joint_loss',joint_loss)
        # TODO: in coarse, should mask the prediction of padded sentences.
        self.joint_pred_labels = self.sf.prediction(name='joint_pred_labels',score=joint_score, Y_atr=attr_pred_labels,graph=self.graph)

    @staticmethod
    def build(config):
        builder = SentiNetBuilder(config)
        dic = builder.build_models(SentimentNet)
        return dic