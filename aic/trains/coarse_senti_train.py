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
np.set_printoptions(threshold=np.inf)
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
        self.train_config['sr_path'] = self.train_config['sr_path']+'model'
        # self.dg is a class
        self.dg = data_feeder
        # self.cl is a class
        self.mt = Metrics(self.train_config)
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
        senti_pred = dic['pred']['senti']
        saver = dic['saver']
        early_stop_count = 0
        best_f1_score = 0
        print('epoch num: ',self.train_config['epoch'])
        pre_A = sess.run(graph.get_collection('A_mat')[0])
        for i in range(self.train_config['epoch']):
            dataset = self.dg.data_generator('train')
            for attr_labels_data, senti_labels_data, sentences_data in dataset:
                data_dict = {'X_data':sentences_data,'Y_att_data':attr_labels_data,
                             'Y_senti_data':senti_labels_data,'keep_prob':self.train_config['keep_prob_lstm']}
                feed_dict = self.generate_feed_dict(graph=graph,gpu_num=gpu_num,data_dict=data_dict)
                _, attr_train_loss, senti_train_loss, attr_pred_data, senti_pred_data \
                    = sess.run([train_step, attr_loss, senti_loss, attr_pred, senti_pred],feed_dict=feed_dict)
                # sa_ls,ra_ls,jf_ls,jc_ls =sess.run([graph.get_collection('sentence_attr_score'),
                #                                    graph.get_collection('review_attr_score'),
                #                                    graph.get_collection('joint_fine_score'),
                #                                    graph.get_collection('joint_coarse_score')],
                #                                  feed_dict=feed_dict)
                # self.mt.report('#########################',self.outf,'report')
                # self.mt.report('sa: %s'%str(sa_ls),self.outf,'report')
                # self.mt.report('ra: %s'%str(ra_ls),self.outf,'report')
                # self.mt.report('jf: %s' % str(jf_ls),self.outf,'report')
                # self.mt.report('jc: %s' % str(jc_ls),self.outf,'report')
                # self.mt.report('attr_train_loss: %.5f'%attr_train_loss,self.outf,'report')
                # self.mt.report('senti_train_loss: %.5f'%senti_train_loss, self.outf, 'report')

            if i % self.train_config['epoch_mod'] == 0:
                self.mt.report('epoch: %d'%i)
                cur_A=sess.run(graph.get_collection('A_mat')[0])
                print('cur_A == pre_A?: ',np.all(np.equal(cur_A,pre_A)))
                pre_A = cur_A
                self.mt.report('\nepoch: %d'%i,self.outf,'report')
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
                    # print('attr_test_loss: ', attr_train_loss)
                    # print('senti_test_loss: ', senti_train_loss)
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
                    TP_data = self.mt.TP(senti_labels_data[:,:-4], senti_pred_data[:,:-4])
                    FP_data = self.mt.FP(senti_labels_data[:,:-4], senti_pred_data[:,:-4])
                    FN_data = self.mt.FN(senti_labels_data[:,:-4], senti_pred_data[:,:-4])
                    senti_TP_vec.append(TP_data)
                    senti_FP_vec.append(FP_data)
                    senti_FN_vec.append(FN_data)
                    senti_loss_vec.append(senti_test_loss)

                TP_vec = np.sum(attr_TP_vec, axis=0)
                FP_vec = np.sum(attr_FP_vec, axis=0)
                FN_vec = np.sum(attr_FN_vec, axis=0)
                loss_value = np.mean(attr_loss_vec)
                self.mt.report('\n#####attribute metrics#####\n',self.outf,'report')
                self.mt.report('Val_loss:%.10f' % loss_value, self.outf, 'report')
                _f1_score = self.mt.calculate_metrics_score(TP_vec=TP_vec, FP_vec=FP_vec, FN_vec=FN_vec,outf=self.outf,id_to_aspect_dic=self.dg.id_to_aspect_dic,mod='attr')

                TP_vec = np.sum(senti_TP_vec, axis=0)
                FP_vec = np.sum(senti_FP_vec, axis=0)
                FN_vec = np.sum(senti_FN_vec, axis=0)
                loss_value = np.mean(senti_loss_vec)
                if dic['test_mod'] !='attr':
                    self.mt.report('\n#####sentiment metrics#####\n', self.outf, 'report')
                    self.mt.report('Val_loss:%.10f' % loss_value, self.outf, 'report')
                    _f1_score = self.mt.calculate_metrics_score(TP_vec=TP_vec, FP_vec=FP_vec, FN_vec=FN_vec,outf=self.outf,id_to_aspect_dic=self.dg.id_to_aspect_dic,mod='senti')


                if best_f1_score >= _f1_score:
                    early_stop_count += 1
                else:
                    early_stop_count = 0
                    best_f1_score = _f1_score
                    saver.save(sess, self.train_config['sr_path'])
                if early_stop_count > self.train_config['early_stop_limit']:
                    break
        saver.save(sess, self.train_config['sr_path'])

    def train(self,model_dic):
        graph = model_dic['graph']
        with graph.as_default():
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
            self.mt.report('attr in training')
            self.mt.report('===========attr============',self.outf,'report')
            dic['train_step'] = model_dic['train_step']['attr']
            dic['loss'] = {'attr':model_dic['loss']['attr'],'senti':model_dic['loss']['joint']}
            dic['pred'] = {'attr':model_dic['pred_labels']['attr'],'senti':model_dic['pred_labels']['joint']}
            dic['test_mod'] = 'attr'
            self.__train__(dic, graph, model_dic['gpu_num'])

            # ##########################
            # train senti (optional)   #
            # ##########################
            self.mt.report('senti in training')
            self.mt.report('===========senti============',self.outf,'report')
            dic['train_step'] = model_dic['train_step']['senti']
            dic['loss'] = {'attr':model_dic['loss']['attr'],'senti':model_dic['loss']['senti']}
            dic['pred'] = {'attr':model_dic['pred_labels']['attr'],'senti':model_dic['pred_labels']['senti']}
            dic['test_mod'] = 'senti'
            self.__train__(dic, graph, model_dic['gpu_num'])

            # ##########################
            # train joint              #
            # ##########################
            self.mt.report('joint in training')
            self.mt.report('===========joint============',self.outf,'report')
            dic['train_step'] = model_dic['train_step']['joint']
            dic['loss'] = {'attr':model_dic['loss']['attr'],'senti':model_dic['loss']['joint']}
            dic['pred'] = {'attr':model_dic['pred_labels']['attr'],'senti':model_dic['pred_labels']['joint']}
            dic['test_mod'] = 'joint'
            self.__train__(dic, graph, model_dic['gpu_num'])