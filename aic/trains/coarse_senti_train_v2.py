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
import pickle

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
                            'attr_sr_path':'/hdd/lujunyu/model/meituan/coarse_nn/model/sentiment/ckpt_reg%s_lr%s_mat%s/' \
                                      % ('1e-5', '0.0001', '3'),
                            'senti_sr_path':'',
                        }
        self.train_config.update(config)
        if not self.train_config['is_restore']:
            dir_ls = ['report_filePath','attr_sr_path','senti_sr_path']
        else:
            dir_ls = ['report_filePath', 'senti_sr_path']
        for name in dir_ls:
            path = Path(self.train_config[name])
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        self.train_config['report_filePath'] = self.train_config['report_filePath'] +'report_reg%s_lr%s_mat%s.info'% \
                                               (str(self.train_config['reg_rate']), str(self.train_config['lr']), str(self.train_config['attribute_mat_size']))
        self.train_config['attr_sr_path'] = self.train_config['attr_sr_path']+'model.ckpt'
        self.train_config['senti_sr_path'] = self.train_config['senti_sr_path'] + 'model.ckpt'
        # self.dg is a class
        self.dg = data_feeder
        # self.cl is a class
        self.mt = Metrics(self.train_config)
        self.outf=open(self.train_config['report_filePath'],'w+')
        # self.analf = open('/datastore/liu121/sentidata2/report/coarse_nn/analysis_reg%s_lr%s_mat%s.info'%
        #                   (str(self.train_config['reg_rate']), str(self.train_config['lr']), str(self.train_config['attribute_mat_size'])),'wb')

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

    def write_to_pkl(self,file,dic):
        pickle.dump(dic,file)
        file.flush()

    def analysis(self,dic,sess,i,feed_dict,result_ls):
        pass

    def get_attr_W(self,sess):
        W_dic={}
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in var_list:
            if var.name.find('attrExtr')>=0:
                W_dic[var.name]=sess.run(var)
            if var.name.find('table')>=0:
                W_dic[var.name] = sess.run(var)
        return W_dic

    def __train__(self, dic, graph, gpu_num,global_step):
        sess = dic['sess']
        train_step = dic['train_step']
        attr_loss = dic['loss']['attr']
        senti_loss = dic['loss']['senti']
        attr_pred = dic['pred']['attr']
        senti_pred = dic['pred']['senti']
        saver = dic['saver']
        early_stop_count = 0
        best_f1_score = 0

        # used to test whether the parameter is changed.
        # if dic['test_mod'] != 'attr':
        #     origin_attrW_dic = self.get_attr_W(sess)

        print('epoch num: ',self.train_config['epoch'])
        for i in range(self.train_config['epoch']):
            self.mt.report('epoch: %d' % i)
            dataset = self.dg.data_generator('train')
            attr_trainLoss_list = []
            senti_trainLoss_list=[]
            for attr_labels_data, senti_labels_data, sentences_data in dataset:
                data_dict = {'X_data': sentences_data, 'Y_att_data': attr_labels_data,
                             'Y_senti_data': senti_labels_data, 'keep_prob': self.train_config['keep_prob_lstm']}
                feed_dict = self.generate_feed_dict(graph=graph, gpu_num=gpu_num, data_dict=data_dict)
                _, attr_train_loss, senti_train_loss, attr_pred_data, senti_pred_data \
                    = sess.run([train_step, attr_loss, senti_loss, attr_pred, senti_pred],feed_dict=feed_dict)
                attr_trainLoss_list.append(attr_train_loss)
                senti_trainLoss_list.append(senti_train_loss)

            if i % self.train_config['epoch_mod'] == 0:
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
                if dic['test_mod'] == 'attr':
                    self.mt.report('\n#####attribute metrics#####\n',self.outf,'report')
                    self.mt.report('Train_loss:%.10f'%np.mean(attr_trainLoss_list),self.outf,'report')
                    self.mt.report('Val_loss:%.10f' % loss_value, self.outf, 'report')
                    _f1_score = self.mt.calculate_metrics_score(TP_vec=TP_vec, FP_vec=FP_vec, FN_vec=FN_vec,outf=self.outf,id_to_aspect_dic=self.dg.id_to_aspect_dic,mod='attr')

                TP_vec = np.sum(senti_TP_vec, axis=0)
                FP_vec = np.sum(senti_FP_vec, axis=0)
                FN_vec = np.sum(senti_FN_vec, axis=0)
                loss_value = np.mean(senti_loss_vec)
                if dic['test_mod'] !='attr':
                    self.mt.report('\n#####sentiment metrics#####\n', self.outf, 'report')
                    self.mt.report('Train_loss:%.10f' % np.mean(senti_trainLoss_list), self.outf, 'report')
                    self.mt.report('Val_loss:%.10f' % loss_value, self.outf, 'report')
                    _f1_score = self.mt.calculate_metrics_score(TP_vec=TP_vec, FP_vec=FP_vec, FN_vec=FN_vec,outf=self.outf,id_to_aspect_dic=self.dg.id_to_aspect_dic,mod='senti')


                if best_f1_score >= _f1_score:
                    early_stop_count += 1
                else:
                    early_stop_count = 0
                    best_f1_score = _f1_score
                    print('save path: %s' % dic['sr_path'])
                    saver.save(sess, dic['sr_path'],global_step=global_step)
                if early_stop_count >= self.train_config['early_stop_limit']:
                    break
        if early_stop_count ==0:
            print('save path: %s' % dic['sr_path'])
            saver.save(sess, dic['sr_path'],global_step=global_step)

    def train(self,model_dic):
        graph = model_dic['graph']
        with graph.as_default():
            table = graph.get_collection('table')[0]
            init = tf.global_variables_initializer()
        table_data = self.dg.table

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(graph=graph, config=config) as sess:
            dic = {'sess': sess, 'saver': model_dic['saver']}
            if not self.train_config['is_restore']:
                sess.run(init, feed_dict={table: table_data})
                # ##############
                # train attr   #
                # ##############
                self.mt.report('attr in training')
                self.mt.report('===========attr============', self.outf, 'report')
                dic['sr_path'] = self.train_config['attr_sr_path']
                dic['train_step'] = model_dic['train_step']['attr']
                dic['loss'] = {'attr': model_dic['loss']['attr'], 'senti': model_dic['loss']['joint']}
                dic['pred'] = {'attr': model_dic['pred_labels']['attr'], 'senti': model_dic['pred_labels']['joint']}
                dic['test_mod'] = 'attr'
                self.__train__(dic, graph, model_dic['gpu_num'], model_dic['global_step'])
            else:
                print('initial path: %s'%self.train_config['initial_path'])
                model_file = tf.train.latest_checkpoint(self.train_config['initial_path'])
                model_dic['saver'].restore(sess, model_file)
                print('restore successful')

            # ##########################
            # train senti (optional)   #
            # ##########################
            self.mt.report('senti in training')
            self.mt.report('===========senti============',self.outf,'report')
            dic['sr_path'] = self.train_config['senti_sr_path']
            dic['train_step'] = model_dic['train_step']['senti']
            dic['loss'] = {'attr':model_dic['loss']['attr'],'senti':model_dic['loss']['senti']}
            dic['pred'] = {'attr':model_dic['pred_labels']['attr'],'senti':model_dic['pred_labels']['senti']}
            # dic['pred'] = {'attr': model_dic['pred_labels']['attr'], 'senti': model_dic['pred_labels']['joint']}
            dic['test_mod'] = 'senti'
            self.__train__(dic, graph, model_dic['gpu_num'],model_dic['global_step'])

            # ##########################
            # train joint              #
            # ##########################
            self.mt.report('joint in training')
            self.mt.report('===========joint============',self.outf,'report')
            dic['train_step'] = model_dic['train_step']['joint']
            dic['loss'] = {'attr':model_dic['loss']['attr'],'senti':model_dic['loss']['joint']}
            dic['pred'] = {'attr':model_dic['pred_labels']['attr'],'senti':model_dic['pred_labels']['joint']}
            dic['test_mod'] = 'joint'
            self.__train__(dic, graph, model_dic['gpu_num'],model_dic['global_step'])