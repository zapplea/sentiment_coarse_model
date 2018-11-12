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
import math

from aic.functions.metrics import Metrics
from aic.functions.tfb_utils import Tfb


class CoarseSentiTrain:
    def __init__(self,config, data_feeder):
        self.train_config ={
                           'epoch': 10,
                           'keep_prob_lstm': 0.5,
                           'top_k_data': -1,
                           'early_stop_limit': 2,
                           'init_model':'/hdd/lujunyu/model/meituan/coarse_nn/model/sentiment/ckpt_reg%s_lr%s_mat%s/' \
                                          % ('1e-5', '0.0001', '3'),
                           'report_filePath':'/hdd/lujunyu/model/meituan/coarse_nn/model/sentiment2/senti_report/',
                            'sr_path':'/hdd/lujunyu/model/meituan/coarse_nn/model/sentiment/ckpt_reg%s_lr%s_mat%s/' \
                                      % ('1e-5', '0.0001', '3'),

                        }
        self.train_config.update(config)
        for name in ['report_filePath','sr_path']:
            path = Path(self.train_config[name])
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        self.train_config['report_filePath'] = self.train_config['report_filePath'] +'report_reg%s_lr%s_mat%s.info'% \
                                               (str(self.train_config['reg_rate']), str(self.train_config['lr']), str(self.train_config['attribute_mat_size']))
        # self.dg is a class
        self.dg = data_feeder
        # self.cl is a class
        self.mt = Metrics()
        self.outf=open(self.train_config['report_filePath'],'w+')

    def generate_feed_dict(self,graph, gpu_num, data_dict):
        feed_dict = {}
        train_mod = math.ceil(data_dict['X_data'].shape[0]/gpu_num)
        for k in range(gpu_num):
            start = k*train_mod
            end = start + train_mod
            feed_dict[graph.get_collection('X')[k]] = data_dict['X_data'][start:end]
            feed_dict[graph.get_collection('Y_att')[k]] = data_dict['Y_att_data'][start:end]
            feed_dict[graph.get_collection('Y_senti')[k]] = data_dict['Y_senti_data'][start:end]
            feed_dict[graph.get_collection('keep_prob_bilstm')[k]] = data_dict['keep_prob']
        return feed_dict

    def __train__(self, dic, graph, gpu_num):
        sess = dic['sess']
        train_step = dic['train_step']
        attr_loss = dic['loss']['attr']
        senti_loss = dic['loss']['senti']
        attr_pred = dic['pred']['attr']
        senti_pred = dic['pred']['attr']
        saver = dic['saver']
        early_stop_count = 0
        best_f1_score = 0

        for i in range(self.train_config['epoch']):
            dataset = self.dg.data_generator('train')
            for attr_labels_data, senti_labels_data, sentences_data in dataset:
                data_dict = {'X_data':sentences_data,'Y_att_data':attr_labels_data,
                             'Y_senti_data':senti_labels_data,'keep_prob':self.train_config['keep_prob_lstm']}
                feed_dict = self.generate_feed_dict(graph=graph,gpu_num=gpu_num,data_dict=data_dict)
                # print('before attr loss')
                # result = sess.run(tf.get_collection('var_attr_loss')[0], feed_dict=feed_dict)
                # print('attr_loss: ', result)
                # print('var senti H')
                # result = sess.run(tf.get_collection('var_senti_H')[0], feed_dict=feed_dict)
                # print(result.shape)
                # print('var attention')
                # result = sess.run(tf.get_collection('var_attention')[0], feed_dict=feed_dict)
                # print(result.shape)
                # print('var attended W:')
                # result = sess.run(tf.get_collection('var_attended_W')[0], feed_dict=feed_dict)
                # print(result.shape)
                # print('var item1:')
                # result = sess.run(tf.get_collection('var_item1')[0], feed_dict=feed_dict)
                # print(result.shape)
                print('fine_score:')
                result = sess.run(tf.get_collection('fine_score')[0],feed_dict=feed_dict)
                print(result.shape)
                print('coarse_score:')
                result = sess.run(tf.get_collection('coarse_score')[0], feed_dict=feed_dict)
                print(result.shape)
                print('joint_fine_score:')
                result = sess.run(tf.get_collection('joint_fine_score')[0], feed_dict=feed_dict)
                print(result.shape)
                print('joint_coarse_score:')
                result = sess.run(tf.get_collection('joint_coarse_score')[0], feed_dict=feed_dict)
                print(result.shape)
                exit()
                _, attr_train_loss, senti_train_loss, attr_pred_data, senti_pred_data \
                    = sess.run([train_step, attr_loss, senti_loss, attr_pred, senti_pred],feed_dict=feed_dict)

            if i % 1 == 0 and i != 0:
                self.mt.report('epoch: %d'%i,self.outf,'report')
                attr_loss_vec = []
                attr_TP_vec = []
                attr_FP_vec = []
                attr_FN_vec = []

                senti_loss_vec = []
                senti_TP_vec = []
                senti_FP_vec = []
                senti_FN_vec = []

                dataset = self.dg.data_generator('val')
                for attr_labels_data, senti_labels_data, sentences_data in dataset:
                    data_dict = {'X_data': sentences_data, 'Y_att_data': attr_labels_data,
                                 'Y_senti_data': senti_labels_data, 'keep_prob': 1.0}
                    feed_dict = self.generate_feed_dict(graph=graph,gpu_num=gpu_num,data_dict=data_dict)
                    attr_test_loss,senti_test_loss, attr_pred_data, senti_pred_data = sess.run(
                        [attr_loss,senti_loss, attr_pred,senti_pred],
                        feed_dict=feed_dict)

                    TP_data = self.mt.TP(attr_labels_data, attr_pred_data)
                    FP_data = self.mt.FP(attr_labels_data, attr_pred_data)
                    FN_data = self.mt.FN(attr_labels_data, attr_pred_data)

                    ###Show test message
                    attr_TP_vec.append(TP_data)
                    attr_FP_vec.append(FP_data)
                    attr_FN_vec.append(FN_data)
                    attr_loss_vec.append(attr_test_loss)

                    senti_labels_data = self.mt.caliberate(senti_labels_data)
                    senti_pred_data = self.mt.caliberate(senti_pred_data)
                    TP_data = self.mt.TP(senti_labels_data[:,:-1,:], senti_pred_data[:,:-1,:])
                    FP_data = self.mt.FP(senti_labels_data[:,:-1,:], senti_pred_data[:,:-1,:])
                    FN_data = self.mt.FN(senti_labels_data[:,:-1,:], senti_pred_data[:,:-1,:])
                    senti_TP_vec.append(TP_data)
                    senti_FP_vec.append(FP_data)
                    senti_FN_vec.append(FN_data)
                    senti_loss_vec.append(senti_test_loss)

                TP_vec = np.concatenate(attr_TP_vec, axis=0)
                FP_vec = np.concatenate(attr_FP_vec, axis=0)
                FN_vec = np.concatenate(attr_FN_vec, axis=0)
                loss_value = np.mean(attr_loss_vec)
                self.mt.report('attribute metrics\n',self.outf,'report')
                self.mt.report('Val_loss:%.10f' % loss_value, self.outf, 'report')
                _f1_score = self.mt.calculate_metrics_score(TP_vec=TP_vec, FP_vec=FP_vec, FN_vec=FN_vec,outf=self.outf,id_to_aspect_dic=self.dg.id_to_aspet_dic)

                TP_vec = np.concatenate(senti_TP_vec, axis=0)
                FP_vec = np.concatenate(senti_FP_vec, axis=0)
                FN_vec = np.concatenate(senti_FN_vec, axis=0)
                loss_value = np.mean(senti_loss_vec)
                if dic['test_mod'] !='attr':
                    self.mt.report('sentiment metrics\n', self.outf, 'report')
                    self.mt.report('Val_loss:%.10f' % loss_value, self.outf, 'report')
                    _f1_score = self.mt.calculate_metrics_score(TP_vec=TP_vec, FP_vec=FP_vec, FN_vec=FN_vec,outf=self.outf,id_to_aspect_dic=self.dg.id_to_aspet_dic)


                if best_f1_score < _f1_score:
                    early_stop_count += 1
                else:
                    early_stop_count = 0
                    best_f1_score = _f1_score
                    saver.save(sess, self.train_config['sr_path'])
                if early_stop_count > self.train_config['early_stop_limit']:
                    break

    def train(self,model_dic):
        graph = model_dic['graph']
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
            # #loss
            # attr_loss = graph.get_collection('atr_loss')[0]
            # senti_loss = graph.get_collection('senti_loss')[0]
            # joint_loss = graph.get_collection('joint_loss')[0]
            #
            # # pred
            # attr_pred = graph.get_collection('atr_pred')[0]
            # senti_pred = graph.get_collection('senti_pred')[0]
            # joint_pred = graph.get_collection('joint_pred')[0]
            #
            # keep_prob_lstm = graph.get_collection('keep_prob_lstm')[0]
            table = graph.get_collection('table')[0]
            init = tf.global_variables_initializer()
        table_data = self.dg.table

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(graph=graph, config=config) as sess:
            sess.run(init, feed_dict={table: table_data})
            # if self.train_config['init_model']:
            #     # model_path = tf.train.latest_checkpoint(self.train_config['init_model'])
            #     # saver.restore(sess, model_path)
            #     print("sucess init %s" % self.train_config['init_model'])
            dic = {'sess': sess, 'saver': model_dic['saver']}

            # ##############
            # train attr   #
            # ##############
            self.mt.report('===========attr============',self.outf,'report')
            dic['train_step'] = model_dic['train_step']['attr']
            dic['loss'] = {'attr':model_dic['loss']['attr'],'senti':model_dic['loss']['joint']}
            dic['pred'] = {'attr':model_dic['pred_labels']['attr'],'senti':model_dic['pred_labels']['joint']}
            dic['test_mod'] = 'attr'

            self.__train__(dic, graph, model_dic['gpu_num'])

            # ##########################
            # train senti (optional)   #
            # ##########################
            # self.mt.report('===========senti============')
            # dic['train_step'] = model_dic['train_step']['senti']
            # dic['loss'] = {'attr':model_dic['loss']['attr'],'senti':model_dic['loss']['senti']}
            # dic['pred'] = {'attr':model_dic['pred_labels']['attr'],'senti':model_dic['pred_labels']['senti']}
            # dic['test_mod'] = 'senti'
            # self.__train__(model_dic, graph, model_dic['gpu_num'])

            # ##########################
            # train joint              #
            # ##########################
            self.mt.report('===========joint============',self.outf,'report')
            dic['train_step'] = model_dic['train_step']['joint']
            dic['loss'] = {'attr':model_dic['loss']['attr'],'senti':model_dic['loss']['joint']}
            dic['pred'] = {'attr':model_dic['pred_labels']['attr'],'senti':model_dic['pred_labels']['joint']}
            dic['test_mod'] = 'joint'
            self.__train__(model_dic, graph, model_dic['gpu_num'])