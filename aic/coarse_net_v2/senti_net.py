import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/code/nlp/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import tensorflow as tf

from aic.functions.coarse_functions import SentimentFunction
from aic.functions.comm_functions import CoarseCommFunction
from aic.functions.coarse_functions import AttributeFunction
from aic.functions.multiGPU_builders import SentiNetBuilder

class SentimentNet:
    def __init__(self,config, graph, table):
        self.nn_config = config
        self.sf = SentimentFunction(self.nn_config)
        self.comm = CoarseCommFunction(self.nn_config)
        self.af = AttributeFunction(self.nn_config)
        self.graph = graph
        self.table = table
        self.reg = {'attr_reg': [], 'senti_reg': []}
        self.classifier()


    def classifier(self):
        # shape = (batch size, max review length, words num)
        X_ids = self.comm.sentences_input(graph=self.graph)
        # Y_att.shape = (batch size, number of attributes)
        Y_att = self.comm.attribute_labels_input(graph=self.graph)
        # TODO: need to regenerate Y_senti to elinminate non-attribute
        # Y_senti.shape = [batch_size, number of attributes, 3]
        Y_senti = self.comm.sentiment_labels_input_v2()
        # shape = (batch size,)
        review_len = self.comm.review_length(X_ids, self.graph)
        # shape = (batch size*max review length, words num)
        reshaped_X_ids = tf.reshape(X_ids, shape=(-1, self.nn_config['words_num']))
        # shape = (batch size*max review length,)
        seq_len = self.comm.sequence_length(reshaped_X_ids, self.graph)
        # shape = (batch size*max review length, words num)
        words_pad_M = self.comm.is_word_padding_input(reshaped_X_ids, self.graph)
        # shape = (batch size*max review length,words num, feature dim)
        X = self.comm.lookup_table(reshaped_X_ids, words_pad_M, self.table, self.graph)
        # the mask is used to mask padded words with -inf
        # shape= (batch size*review length,max words num)
        mask = self.comm.mask_for_pad_in_score(X_ids, self.graph)
        # #################### #
        # attribute extraction #
        # #################### #
        with tf.variable_scope('attrExtr', reuse=tf.AUTO_REUSE):
            # H.shape = (batch size*max review length, max_time, cell size)
            pre_H = X
            score_ls = []
            sentence_ls = [X]
            for i in range(self.nn_config['CoarseSenti_v2']['bilstm']['n_layers']):
                with tf.variable_scope('bilstm_layer_'+str(i),reuse=tf.AUTO_REUSE):
                    scope_name = tf.get_default_graph().get_name_scope()
                    scope_name_ls = scope_name.split('/')
                    scope_name_ls[0] = 'sentiment'
                    scope_name = '/'.join(scope_name_ls)
                    # attr_H.shape = (batch size*max review length, max time step, lstm cell size)
                    attr_H = self.comm.sentence_bilstm('attr_reg',
                                                       pre_H,
                                                       seq_len,
                                                       reg=self.reg,
                                                       graph=self.graph,
                                                       scope_name=scope_name)
                    sentence_ls.append(attr_H)
                    pre_H = attr_H
                with tf.variable_scope('attr_mat_'+str(i), reuse=tf.AUTO_REUSE):
                    A, o = self.af.attribute_mat(self.reg,self.graph)
                    tf.add_to_collection('A_mat',A)
                    tf.add_to_collection('o_mat',o)
                    # A.shape = (batch size*max review length, words num, attributes number, attribute dim)
                    A_lstm = self.af.words_attribute_mat2vec(attr_H, A, self.graph)
                    tf.add_to_collection('A_lstm', A_lstm)
                    o_lstm = self.af.words_nonattribute_mat2vec(attr_H, o, self.graph)
                    A_lstm = A_lstm - o_lstm
                    A_e = self.af.words_attribute_mat2vec(X, A, self.graph)
                    tf.add_to_collection('A_e', A_e)
                    o_e = self.af.words_nonattribute_mat2vec(X, o, self.graph)
                    A_e = A_e - o_e
                with tf.variable_scope('score_'+str(i), reuse=tf.AUTO_REUSE):
                    # score.shape = (batch size*max review length, attributes num, words num)
                    score_lstm = self.af.sentence_score(A_lstm, attr_H, mask, self.graph)
                    # score.shape = (batch size*max review length, attributes num, words num)
                    score_e = self.af.sentence_score(A_e, X, mask, self.graph)
                    # score.shape = (batch size*max review length, attributes num, words num)
                    score = tf.add(score_lstm, score_e)
                    # score_ls.shape = (bilstm n_layers,batch size*max review length, attributes num, words num)
                    score_ls.append(score)
            tf.add_to_collection('score_ls',score_ls)
            # Done: need to careful the padded sentence, there will be nan because all exp('-inf')=0 then 0/0 = nan
            # attention.shape = (batch size*max review length, attributes num, words num)
            sentence_attention = self.af.sentence_attention(score_ls)
            tf.add_to_collection('sentence_attention',sentence_attention)
            # sentence_repr.shape = (batch size*max review length, attributes num, n_layers*lstm cell size)
            # for the padded sentence, all words have the same attention. for patially paded sentence, the padded
            # words' attention will be 0.
            attr_sentence_repr,n_layers = self.af.sentence_repr(sentence_attention,[sentence_ls[0],sentence_ls[-1]])
            tf.add_to_collection('attr_sentence_repr',attr_sentence_repr)

        # #################### #
        # sentiment extraction #
        # #################### #
        # sentiment lstm
        with tf.variable_scope('sentiExtr', reuse= tf.AUTO_REUSE):
            pre_H = X
            # shape = (n_layers, batch size * max review length, max_time, cell size)
            for i in range(self.nn_config['CoarseSenti_v2']['bilstm']['n_layers']):
                with tf.variable_scope('bilstm_layer_' + str(i), reuse=tf.AUTO_REUSE):
                    scope_name = tf.get_default_graph().get_name_scope()
                    scope_name_ls = scope_name.split('/')
                    scope_name_ls[0]='sentiment'
                    scope_name='/'.join(scope_name_ls)
                    # H.shape = (batch size * max review length, max_time, cell size)
                    senti_H = self.comm.sentence_bilstm('senti_reg',
                                                        pre_H,
                                                        seq_len,
                                                        self.reg,
                                                        graph=self.graph,
                                                        scope_name=scope_name)
                    pre_H = senti_H
            # need to mask padded sentence and words in max_pooling.
            # padded sentence is (-inf, -inf, ..., -inf)
            # shape = (batch size * max review length, cell size)
            senti_sentence_repr=self.sf.max_pooling(senti_H,mask)
            tf.add_to_collection('senti_sentence_repr',senti_sentence_repr)

        with tf.variable_scope('document',reuse=tf.AUTO_REUSE):
            # shape = (attributes num, context num, sentence_repr dim)
            Z_mat = self.comm.context_matrix(self.reg,n_layers)
            tf.add_to_collection('Z_mat',Z_mat)
            # shape = (attributes num, batch size, context num, max review length)
            document_attention = self.comm.document_attention(Z_mat, attr_sentence_repr,mask)
            tf.add_to_collection('document_attention_ls',document_attention)
            # shape = (attributes num, batch size, context num*n_layers*lstm cell size)
            attr_D_repr = self.af.attr_document_repr(document_attention, attr_sentence_repr,n_layers)
            tf.add_to_collection('attr_D_repr',attr_D_repr)
            # shape = (attributes num, batch size, context num*lstm cell size)
            senti_D_repr = self.sf.senti_document_repr(document_attention, senti_sentence_repr)
            tf.add_to_collection('senti_D_repr',senti_D_repr)

            # shape = (batch size, attributes num)
            attr_score = self.af.review_score_v2(attr_D_repr,n_layers,self.reg)
            tf.add_to_collection('attr_score',attr_score)
            # shape = (batch size, attributes num, sentiment num)
            senti_score = self.sf.review_score_v2(senti_D_repr,self.reg)
            tf.add_to_collection('senti_score',senti_score)
            #
            reg_list = []
            for reg in self.reg['attr_reg']:
                reg_list.append(reg)

            tf.add_to_collection('attr_reg',reg_list)
            tf.add_to_collection('attr_reg_sum', tf.reduce_sum(reg_list))
            attr_loss = self.af.sigmoid_loss('attr_loss', attr_score, Y_att, reg_list, self.graph)

            attr_pred_labels = self.af.prediction('attr_pred_labels', attr_score, self.graph)

            # TODO: need to add attr pred to senti loss
            reg_list = []
            for reg in self.reg['senti_reg']:
                reg_list.append(reg)

            # pure senti
            # masked senti score
            # shape = (batch size, attributes num, sentiment num)
            senti_score = self.sf.mask_senti_score(senti_score,attr_label=Y_att)
            tf.add_to_collection('masked_senti_score', senti_score)
            senti_loss = self.sf.softmax_loss(name='senti_loss',labels=Y_senti, logits=senti_score, reg_list=reg_list,
                                              graph=self.graph)
            senti_pred_labels = self.sf.prediction(name='senti_pred_labels', score=senti_score, Y_att=Y_att,
                                                   graph=self.graph)

            # joint senti
            # shape = (batch size, attributes num, sentiment num)
            senti_score = self.sf.mask_senti_score(senti_score, attr_label=attr_pred_labels)
            joint_loss = self.sf.softmax_loss(name='joint_loss', labels=Y_senti, logits=senti_score, reg_list=reg_list,
                                              graph=self.graph)
            joint_pred_labels = self.sf.prediction(name='joint_pred_labels', score=senti_score, Y_att=attr_pred_labels,
                                                   graph=self.graph)

    @staticmethod
    def build(config):
        builder = SentiNetBuilder(config)
        dic = builder.build_models(SentimentNet)
        return dic