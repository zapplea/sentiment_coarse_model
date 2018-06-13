import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import tensorflow as tf
import numpy as np
from sentiment.functions.sentiment_function.sentiment_function import SentiFunction

class Transfer:
    def __init__(self, coarse_nn_config, coarse_data_generator):
        self.coarse_nn_config = coarse_nn_config
        self.coarse_data_generator = coarse_data_generator
        self.sf = SentiFunction(coarse_nn_config)

    def softmax(self, score):
        """

        :param score: (sentences number, attributes num)
        :return: 
        """
        avg_score = tf.reduce_mean(score, axis=0)
        index = tf.argmax(tf.nn.softmax(avg_score, axis=0))
        return index

    def gather_score(self,score,Y_senti):
        """
        
        :param score: 
        :param Y_senti: 
        :return: 
        """
        tf.argmax(Y_senti,axis=)

    def transfer(self, coarse_model,fine_dg):
        graph, saver = coarse_model.classifier()
        score = graph.get_collection('senti_score')[0]
        Y_att=graph.get_collection('check')[0]
        # mask the situation when attribute doesn't appear
        mask = tf.tile(tf.expand_dims(Y_att, axis=2), multiples=[1, 1, 3])
        # score.shape = (batch size, number of attributes+1,3)
        score = tf.multiply(tf.reshape(score, shape=(-1, self.coarse_nn_config['attributes_num'] + 1, 3)), mask)
        Y_senti=graph.get_collection('Y_senti')[0]


        X = graph.get_collection('X')[0]
        keep_prob_lstm = graph.get_collection('keep_prob_lstm')[0]
        bilstm_fw_kernel = graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')
        bilstm_fw_bias = graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/bias:0')
        bilstm_bw_kernel =graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')
        bilstm_bw_bias = graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/bias:0')
        # attribute variables
        if self.coarse_nn_config['is_mat']:
            A = graph.get_collection('A_mat')[0]
            O = graph.get_collection('o_mat')[0]
        else:
            A = graph.get_collection('A_vec')[0]
            O = graph.get_collection('o_vec')[0]
        # sentiment variables
        # shape = (3*numbers of normal sentiment prototype + attributes_numbers*attribute specific sentiment prototypes)
        W=graph.get_collection('W')[0]
        # sentiment extractors for all (yi,ai)
        # extors_mat.shape = (3*attributes number+3, sentiment prototypes number, sentiment dim)
        extors_mat = self.sf.senti_extors_mat(graph)
        # W.shape=(number of attributes*3+3, size of original W); shape of original W =(3*normal sentiment prototypes + attribute number * attribute sentiment prototypes, sentiment dim)
        W = tf.multiply(extors_mat, W)

        index = self.softmax(score)
        # X_data.shape = (fine grained attributes number, number of sentences,1,words num)
        X_data_list = self.coarse_data_generator.fine_sentences(fine_dg.train_attribute_ground_truth,fine_dg.train_sentence_ground_truth)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        index_list = []
        initializer_A_data = []
        with tf.Session(graph=graph, config=config) as sess:
            model_file = tf.train.latest_checkpoint(self.coarse_nn_config['sr_path'])
            saver.restore(sess, model_file)
            for X_data in X_data_list:
                index_data,score_data = sess.run([index,score], feed_dict={X: X_data,keep_prob_lstm:1.0})
                print(index_data, np.mean(score_data, axis=0))
                index_list.append(index_data)

            A_data, initializer_O_data = sess.run([A, O])
            for index_data in index_list:
                initializer_A_data.append(A_data[index_data.astype('int32')])
            bilstm_fw_kernel_data,bilstm_fw_bias_data=sess.run([bilstm_fw_kernel,bilstm_fw_bias])
            bilstm_bw_kernel_data, bilstm_bw_bias_data = sess.run([bilstm_bw_kernel, bilstm_bw_bias])
        init_data = {'init_A': initializer_A_data, 'init_O': initializer_O_data,
                     'init_bilstm_fw_kernel': bilstm_fw_kernel_data, 'init_bilstm_fw_bias': bilstm_fw_bias_data,
                     'init_bilstm_bw_kernel':bilstm_bw_kernel_data,'init_bilstm_bw_bias':bilstm_bw_bias_data}
        return init_data