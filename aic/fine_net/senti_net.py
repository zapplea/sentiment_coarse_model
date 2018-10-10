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

class SentimentNet:
    def __init__(self,config):
        self.nn_config = {
            'words_num': 40,
            'lstm_cell_size': 300,
            'word_dim': 300,
            'attribute_dim': 300,
            'lookup_table_words_num': 34934,  # 34934,2074276 for Chinese word embedding
            'padding_word_index': 34933,  # 34933,the index of #PAD# in word embeddings list
            'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
            'attributes_num': 12,
            'batch_size': 200,
            'atr_pred_threshold': 0,
            'normal_senti_prototype_num': None,
            'attribute_senti_prototype_num':None,
            'sentiment_dim':300,
            'rps_num':None,
            'rps_dim':None,
            'reg_rate': 1E-5,
            'lr': 1E-4,
        }
        self.nn_config.update(config)
        self.sf = SentimentFunction(self.nn_config)
        self.comm = FineCommFunction(self.nn_config)
        self.af = AttributeFunction(self.nn_config)

    def classifier(self):
        graph = tf.Graph()
        with graph.as_default():
            X_ids = self.comm.sentences_input(graph)
            # shape = (batch size, number of attributes, 3)
            Y_att = self.comm.attribute_labels_input(graph)
            # #################### #
            # attribute extraction #
            # #################### #
            words_pad_M = self.comm.is_word_padding_input(X_ids, graph)
            X = self.comm.lookup_table(X_ids, words_pad_M, graph)
            # lstm
            with tf.variable_scope('sentence_bilstm'):
                seq_len = self.comm.sequence_length(X_ids, graph)
                # H.shape = (batch size, max_time, cell size)
                attr_H = self.comm.sentence_bilstm(X, seq_len, graph=graph)
            A, o = self.af.attribute_mat(graph)
            # A.shape = (batch size, words num, attributes number, attribute dim)
            A_lstm = self.af.words_attribute_mat2vec(attr_H, A, graph)
            o_lstm = self.af.words_nonattribute_mat2vec(attr_H, o, graph)
            A_lstm = A_lstm - o_lstm
            A_e = self.af.words_attribute_mat2vec(X, A, graph)
            o_e = self.af.words_nonattribute_mat2vec(X, o, graph)
            A_e = A_e - o_e

            mask = self.comm.mask_for_pad_in_score(X_ids, graph)
            score_lstm = self.af.score(A_lstm, attr_H, mask, graph)
            # score.shape = (batch size, attributes num, words num)
            score_e = self.af.score(A_e, X, mask, graph)
            # score.shape = (batch size, attributes num, words num)
            score = tf.add(score_lstm, score_e)
            graph.add_to_collection('score_pre', score)

            # score.shape = (batch size, attributes num)
            score = tf.reduce_max(score, axis=2)
            graph.add_to_collection('attr_score', score)
            attr_loss = self.af.sigmoid_loss(score, Y_att, graph)
            attr_pred = self.af.prediction(score, graph)
            opt = self.comm.optimizer(attr_loss, graph=graph)
            graph.add_to_collection('attr_opt',opt)


            # #################### #
            # sentiment extraction #
            # #################### #
            # sentiment lstm
            with tf.variable_scope('senti_sentence_bilstm'):
                # H.shape = (batch size, max_time, cell size)
                senti_H = self.comm.sentence_bilstm(X, seq_len, graph=graph)
            # Y_senti.shape = [batch_size, number of attributes + 1, 3]
            Y_senti = self.comm.sentiment_labels_input(graph=graph)
            # sentiment expression prototypes matrix
            # shape = (3*numbers of normal sentiment prototype + attributes_numbers*attribute specific sentiment prototypes)
            W = self.sf.sentiment_matrix(graph)
            # sentiment extractors for all (yi,ai)
            # extors_mat.shape = (3*attributes number+3, sentiment prototypes number, sentiment dim)
            extors_mat = self.sf.senti_extors_mat(graph)
            # extors_mask_mat.shape = (3*attributes number+3, sentiment prototypes number)
            extors_mask_mat = self.sf.extors_mask(extors=extors_mat, graph=graph)
            # relative position matrix
            V = self.sf.relative_pos_matrix(graph)
            # relative position ids
            # shape = (number of words, number of words)
            rp_ids = self.sf.relative_pos_ids(graph)
            beta = self.sf.beta(graph)
            # extract sentiment expression corresponding to sentiment and attribute from W for all attributes
            # W.shape=(number of attributes*3+3, size of original W); shape of original W =(3*normal sentiment prototypes + attribute number * attribute sentiment prototypes, sentiment dim)
            W = tf.multiply(extors_mat, W)
            # shape = (batch size,number of words, 3+3*attributes number, number of sentiment prototypes)
            attention = self.sf.sentiment_attention(senti_H, W, extors_mask_mat, graph)
            # attended_W.shape = (batch size,number of words, 3+3*attributes number, sentiment dim)
            attended_W = self.sf.attended_sentiment(W, attention, graph)
            # shape = (batch size,number of words, 3+3*attributes number)
            item1 = self.sf.item1(attended_W, senti_H, graph)
            # A_dist.shape = (batch size, number of attributes+1, wrods number)
            if self.nn_config['is_mat']:
                A = tf.concat([A, o], axis=0)
                # A.shape = shape = (batch size, number of words, number of attributes + 1, attribute dim(=lstm cell dim))
                A = self.sf.words_attribute_mat2vec(H=attr_H, A_mat=A, graph=graph)

            A_dist = self.sf.attribute_distribution(A=A, H=attr_H, graph=graph)
            # A_Vi.shape = (batch size, number of attributes+1, number of words, relative position dim)
            A_Vi = self.sf.rd_Vi(A_dist=A_dist, V=V, rp_ids=rp_ids, graph=graph)
            # item2.shape=(batch size, number of attributes+1, number of words)
            item2 = tf.reduce_sum(tf.multiply(A_Vi, beta), axis=3)
            # mask for score to eliminate the influence of padded word
            mask = self.sf.mask_for_pad_in_score(X_ids, graph)
            # senti_socre.shape = (batch size, 3*number of attributes+3, words num)
            score = self.sf.score(item1, item2, mask, graph)
            # score.shape = (batch size, 3*number of attributes+3)
            score = tf.reduce_max(score, axis=2)

            # pure senti loss
            # mask the situation when attribute doesn't appear
            Y_att = self.sf.expand_attr_labels(Y_att)
            mask = tf.tile(tf.expand_dims(Y_att, axis=2), multiples=[1, 1, 3])
            # score.shape = (batch size, number of attributes+1,3)
            pure_score = tf.multiply(tf.reshape(score, shape=(-1, self.nn_config['attributes_num'] + 1, 3)), mask)
            graph.add_to_collection('pure_senti_score', pure_score)
            # softmax loss
            # TODO: check reg
            senti_loss = self.sf.softmax_loss(labels=Y_senti, logits=pure_score, graph=graph)
            opt = self.comm.optimizer(senti_loss, graph)
            graph.add_to_collection('senti_opt',opt)
            senti_pred = self.sf.prediction(name='senti_pred',score=pure_score, Y_atr=Y_att, graph=graph)

            # joint loss
            # mask the situation when attribute doesn't appear
            attr_pred = self.sf.expand_attr_labels(attr_pred)
            mask = tf.tile(tf.expand_dims(attr_pred, axis=2), multiples=[1, 1, 3])
            # score.shape = (batch size, number of attributes+1,3)
            joint_score = tf.multiply(tf.reshape(score, shape=(-1, self.nn_config['attributes_num'] + 1, 3)), mask)
            graph.add_to_collection('joint_senti_score', joint_score)
            # softmax loss
            # TODO: check reg
            senti_loss = self.sf.softmax_loss(labels=Y_senti, logits=joint_score, graph=graph)
            joint_opt = self.sf.joint_optimizer(senti_loss, attr_loss, graph)
            graph.add_to_collection('joint_opt',joint_opt)
            # TODO: in coarse, should mask the prediction of padded sentences.
            joint_pred = self.sf.prediction(name='joint_pred',score=joint_score, Y_atr=attr_pred, graph=graph)
            saver = tf.train.Saver()

            return graph, saver





















