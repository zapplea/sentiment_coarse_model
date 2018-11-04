import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
elif getpass.getuser() == "lizhou":
    sys.path.append('/media/data2tb4/yibing2/sentiment_coarse_model/')

import tensorflow as tf
import numpy as np
from pathlib import Path

from aic.functions.metrics import Metrics
from aic.functions.tfb_utils import Tfb

class TransSentiTrain:
    def __init__(self,config,data_feeder):
        self.train_config = {
                            'epoch': 10,
                            'reg_rate': 1E-5,
                            'lr': 1E-4,
                            'keep_prob_lstm': 0.5,
                            'top_k_data': -1,
                            'early_stop_limit': 2,
                            'report_filePath': '/datastore/liu121/sentidata2/report/transfer_nn/',
                            'sr_path': '/datastore/liu121/sentidata2/result/fine_nn/model/ckpt_reg%s_lr%s_mat%s/' \
                                       % ('1e-5', '0.0001', '3'),
                            'initial_file_path':'/datastore/liu121/sentidata2/result/fine_nn/ckpt_reg%s_lr%s_mat%s/' \
                                           %('1e-5','0.0001','3'),
                            }
        for name in ['report_filePath','sr_path']:
            path = Path(self.train_config[name])
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        self.train_config['report_filePath'] = self.train_config['report_filePath'] +'report_reg%s_lr%s_mat%s.info'% ('1e-5', '0.0001', '3')
        self.train_config.update(config)

        # self.dg is a class
        self.dg = data_feeder
        # self.cl is a class
        self.mt = Metrics()

    def transfer(self,model_dic):
        graph = model_dic['graph']
        saver = model_dic['saver']
        with graph.as_default():
            bilstm_fw_kernel = graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')
            bilstm_fw_bias = graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/bias:0')
            bilstm_bw_kernel = graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')
            bilstm_bw_bias = graph.get_tensor_by_name('sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/bias:0')
            A = graph.get_tensor_by_name('A_mat')
            O = graph.get_tensor_by_name('o_mat')
            table = graph.get_tensor_by_name('table')
            senti_matrix= graph.get_tensor_by_name('senti_mat')
            relpos_matrix = graph.get_tensor_by_name('relative_pos')
            beta = graph.get_tensor_by_name('beta')

        with graph.device('/gpu:0'):
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(graph=graph, config=config) as sess:
                model_file = tf.train.latest_checkpoint(self.train_config['initial_file_path'])
                saver.restore(sess, model_file)
                table_data, A_data, O_data, bilstm_fw_kernel_data, bilstm_fw_bias_data, bilstm_bw_kernel_data, bilstm_bw_bias_data,senti_matrix_data,relpos_matrix_data,beta_data=\
                    sess.run([table,A,O,bilstm_fw_kernel,bilstm_fw_bias,bilstm_bw_kernel,bilstm_bw_bias,senti_matrix,relpos_matrix,beta])
                init={'table':table_data,'A':A_data,'O':O_data,
                      'bilstm_fw_kernel':bilstm_fw_kernel_data,
                      'bilstm_fw_bias':bilstm_fw_bias_data,
                      'bilstm_bw_kernel':bilstm_bw_kernel_data,
                      'bilstm_bw_bias':bilstm_bw_bias_data,
                      'senti_matrix':senti_matrix_data,
                      'relpos_matrix':relpos_matrix_data,
                      'beta':beta_data}
        return init

    def __train__(self, dic, graph, gpu_num):
        sess = dic['sess']
        train_step = dic['train_step']
        loss = dic['loss']
        pred = dic['pred']
        saver = dic['saver']
        early_stop_count = 0
        best_f1_score = 0

        for i in range(self.train_config['epoch']):
            dataset = self.dg.data_generator('train')
            for attr_labels_data, senti_labels_data, sentences_data in dataset:
                data_dict = {'X_data':sentences_data,'Y_att_data':attr_labels_data,
                             'Y_senti_data':senti_labels_data,'keep_prob':self.train_config['keep_prob_lstm']}
                feed_dict = self.generate_feed_dict(graph=graph,gpu_num=gpu_num,data_dict=data_dict)
                _, train_loss, pred_data, \
                    = sess.run([train_step, loss, pred],feed_dict=feed_dict)

            if i % 1 == 0 and i != 0:
                loss_vec = []
                TP_vec = []
                FP_vec = []
                FN_vec = []
                dataset = self.dg.data_generator('val')
                for attr_labels_data, senti_labels_data, sentences_data in dataset:
                    data_dict = {'X_data': sentences_data, 'Y_att_data': attr_labels_data,
                                 'Y_senti_data': senti_labels_data, 'keep_prob': 1.0}
                    feed_dict = self.generate_feed_dict(graph=graph,gpu_num=gpu_num,data_dict=data_dict)
                    test_loss, pred_data = sess.run(
                        [loss, pred],
                        feed_dict=feed_dict)
                    if dic['test_mod'] == 'attr':
                        TP_data = self.mt.TP(attr_labels_data, pred_data)
                        FP_data = self.mt.FP(attr_labels_data, pred_data)
                        FN_data = self.mt.FN(attr_labels_data, pred_data)
                    else:
                        d0 = senti_labels_data.shape[0]
                        d1 = senti_labels_data.shape[1]
                        d2 = senti_labels_data.shape[2]
                        senti_labels_data = np.reshape(senti_labels_data,newshape=(d0,d1*d2))
                        pred_data = np.reshape(pred_data,newshape=(d0,d1*d2))
                        TP_data = self.mt.TP(senti_labels_data, pred_data)
                        FP_data = self.mt.FP(senti_labels_data, pred_data)
                        FN_data = self.mt.FN(senti_labels_data, pred_data)
                    ###Show test message
                    TP_vec.append(TP_data)
                    FP_vec.append(FP_data)
                    FN_vec.append(FN_data)
                    loss_vec.append(test_loss)

                TP_vec = np.concatenate(TP_vec, axis=0)
                FP_vec = np.concatenate(FP_vec, axis=0)
                FN_vec = np.concatenate(FN_vec, axis=0)
                print('Val_loss:%.10f' % np.mean(loss_vec))

                _precision = self.mt.precision(TP_vec, FP_vec, 'macro')
                _recall = self.mt.recall(TP_vec, FN_vec, 'macro')
                _f1_score = self.mt.f1_score(_precision, _recall, 'macro')
                print('Macro F1 score:', _f1_score, ' Macro precision:', np.mean(_precision),
                      ' Macro recall:', np.mean(_recall))

                if best_f1_score < _f1_score:
                    early_stop_count += 1
                else:
                    early_stop_count = 0
                    best_f1_score = _f1_score
                    saver.save(sess, self.train_config['sr_path'])
                if early_stop_count > self.train_config['early_stop_limit']:
                    break

    def train(self,model_dic, init_data):
        graph = model_dic['graph']
        saver = model_dic['saver']

        with graph.as_default():
            # # input
            # X = graph.get_collection('X')[0]
            # # labels
            # Y_att = graph.get_collection('Y_att')[0]
            # Y_senti = graph.get_collection('Y_senti')[0]
            # # train_step
            # attr_train_step = graph.get_collection('attr_opt')[0]
            # senti_train_step = graph.get_collection('senti_opt')[0]
            # joint_train_step = graph.get_collection('joint_opt')[0]
            # #
            # table = graph.get_collection('table')[0]
            # # loss
            # attr_loss = graph.get_collection('atr_loss')[0]
            # senti_loss = graph.get_collection('senti_loss')[0]
            # joint_loss = graph.get_collection('joint_loss')[0]

            # # pred
            # attr_pred = graph.get_collection('atr_pred')[0]
            # senti_pred = graph.get_collection('senti_pred')[0]

            bilstm_fw_kernel = graph.get_tensor_by_name('sentiment/sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')
            bilstm_fw_bias = graph.get_tensor_by_name('sentiment/sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/bias:0')
            bilstm_bw_kernel = graph.get_tensor_by_name('sentiment/sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')
            bilstm_bw_bias = graph.get_tensor_by_name('sentiment/sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/bias:0')
            A = graph.get_tensor_by_name('sentiment/A_mat')
            O = graph.get_tensor_by_name('sentiment/o_mat')
            table = graph.get_tensor_by_name('table')
            senti_mat = graph.get_tensor_by_name('senti_mat')
            relpos_mat = graph.get_tensor_by_name('relative_pos')
            beta = graph.get_tensor_by_name('beta')

            # attribute function
            init = tf.global_variables_initializer()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(graph=graph, config=config) as sess:
            sess.run(init, feed_dict={table: init_data['table']})
            A.load(init_data['A'], sess)
            O.load(init_data['O'], sess)
            bilstm_fw_kernel.load(init_data['bilstm_fw_kernel'], sess)
            bilstm_fw_bias.load(init_data['bilstm_fw_bias'], sess)
            bilstm_bw_kernel.load(init_data['bilstm_bw_kernel'], sess)
            bilstm_bw_bias.load(init_data['bilstm_bw_bias'], sess)
            senti_mat.load(init_data['senti_matrix'],sess)
            relpos_mat.load(init_data['relpos_matrix'],sess)
            beta.load(init_data['beta'],sess)
            dic = {'sess': sess, 'saver': model_dic['saver']}

            # ##############
            # train attr   #
            # ##############
            dic['train_step'] = model_dic['train_step']['attr']
            dic['loss'] = model_dic['loss']['attr']
            dic['pred'] = model_dic['pred_labels']['attr']
            dic['test_mod'] = 'attr'

            self.__train__(dic, graph, model_dic['gpu_num'])

            # ##########################
            # train senti (optional)   #
            # ##########################
            dic['train_step'] = model_dic['train_step']['senti']
            dic['loss'] = model_dic['loss']['senti']
            dic['pred'] = model_dic['pred_labels']['senti']
            dic['test_mod'] = 'senti'
            self.__train__(model_dic, graph, model_dic['gpu_num'])

            # ##########################
            # train joint              #
            # ##########################
            dic['train_step'] = model_dic['train_step']['joint']
            dic['loss'] = model_dic['loss']['joint']
            dic['pred'] = model_dic['pred_labels']['joint']
            dic['test_mod'] = 'joint'
            self.__train__(model_dic, graph, model_dic['gpu_num'])