import tensorflow as tf
import numpy as np


class Transfer:
    def __init__(self, coarse_nn_config, coarse_data_generator):
        self.coarse_nn_config = coarse_nn_config
        self.coarse_data_generator = coarse_data_generator

    def softmax(self, score):
        """

        :param score: (sentences number, attributes num)
        :return: 
        """
        avg_score = tf.reduce_mean(score, axis=0)
        index = tf.argmax(tf.nn.softmax(avg_score, axis=0))
        return index

    def transfer(self, coarse_model,fine_dg):
        graph, saver = coarse_model.classifier()
        score = graph.get_collection('score')[0]
        X = graph.get_collection('X')[0]
        keep_prob_lstm = graph.get_collection('keep_prob_lstm')[0]
        bilstm_fw_kernel = graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')
        bilstm_fw_bias = graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/bias:0')
        bilstm_bw_kernel =graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')
        bilstm_bw_bias = graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/bias:0')
        if self.coarse_nn_config['is_mat']:
            A = graph.get_collection('A_mat')[0]
            O = graph.get_collection('o_mat')[0]
        else:
            A = graph.get_collection('A_vec')[0]
            O = graph.get_collection('o_vec')[0]

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
                     'init_bilstm_bw_kernel':bilstm_bw_kernel_data,'init_bilstm_bw_bias':bilstm_bw_bias_data,
                     'coarse_A':A}
        return init_data