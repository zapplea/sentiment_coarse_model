import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import tensorflow as tf

from aic.functions.coarse_functions import AttributeFunction
from aic.functions.comm_functions import CoarseCommFunction


class AttributeNet:
    def __init__(self,config):
        self.nn_config = {
            'words_num': 40,
            'lstm_cell_size': 300,
            'word_dim': 300,
            'attribute_dim': 300,
            'lookup_table_words_num': 34934,  # 34934,2074276 for Chinese word embedding
            'padding_word_index': 34933,  # 34933,the index of #PAD# in word embeddings list
            'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
            'attributes_num': 12, # fine attributes number
            'coarse_attributes_num':22,
            'batch_size': 200,
            'atr_pred_threshold': 0,
            'review_atr_pred_threshold':0.5,
            'max_review_len':20
        }
        self.nn_config.update(config)
        self.af = AttributeFunction(self.nn_config)
        self.comm = CoarseCommFunction(self.nn_config)

    def classifier(self):
        graph = tf.Graph()
        with graph.as_default():
            # shape = (batch size, max review length, words num)
            X_ids = self.comm.sentences_input(graph=graph)
            # shape = (batch size,)
            review_len = self.comm.review_length(X_ids, graph)
            # shape = (batch size*max review length, words num)
            reshaped_X_ids = tf.reshape(X_ids,shape=(-1,self.nn_config['words_num']))
            # shape = (batch size*max review length,)
            seq_len = self.comm.sequence_length(reshaped_X_ids, graph)
            # shape = (batch size*max review length, words num)
            words_pad_M = self.comm.is_word_padding_input(reshaped_X_ids, graph)
            # shape = (batch size*max review length,words num, feature dim)
            X = self.comm.lookup_table(reshaped_X_ids, words_pad_M, graph)

            # lstm
            with tf.variable_scope('sentence_bilstm'):
                # H.shape = (batch size*max review length, max_time, cell size)
                H = self.comm.sentence_bilstm(X, seq_len, graph=graph)
            # Y_att.shape = (batch size, number of attributes)
            Y_att = self.comm.attribute_labels_input(graph=graph)

            A, o = self.af.attribute_mat(graph)
            # A.shape = (batch size*max review length, words num, attributes number, attribute dim)
            A_lstm = self.af.words_attribute_mat2vec(H, A, graph)
            o_lstm = self.af.words_nonattribute_mat2vec(H, o, graph)
            A_lstm = A_lstm - o_lstm
            A_e = self.af.words_attribute_mat2vec(X, A, graph)
            o_e = self.af.words_nonattribute_mat2vec(X, o, graph)
            A_e = A_e - o_e

            mask = self.comm.mask_for_pad_in_score(X_ids, graph)
            score_lstm = self.af.sentence_score(A_lstm, H, mask, graph)
            # score.shape = (batch size*max review length, attributes num, words num)
            score_e = self.af.sentence_score(A_e, X, mask, graph)
            # score.shape = (batch size*max review length, attributes num, words num)
            score = tf.add(score_lstm, score_e)
            graph.add_to_collection('score_pre', score)

            # score.shape = (batch size*max review length, attributes num)
            score = tf.reduce_max(score, axis=2)
            graph.add_to_collection('score', score)
            # shape = (batch size*max review length, attributes num)
            sentence_prob = self.af.sentence_sigmoid(score,graph)
            #shape = (batch size*max review length, coarse attributes num)
            mask=self.af.review_mask(X_ids)
            # shape = (batch size, coarse_attributes_num)
            review_score = self.af.review_score(sentence_prob,mask,review_len,graph)
            pred = self.af.prediction(review_score,graph)
            loss = self.af.sigmoid_loss(review_score,Y_att,graph)


