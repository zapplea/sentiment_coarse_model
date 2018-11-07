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
                           'epoch': 100,
                           'keep_prob_lstm': 0.5,
                           'top_k_data': -1,
                           'early_stop_limit': 2,
                           'init_model':'/hdd/lujunyu/model/meituan/coarse_nn/model/sentiment/ckpt_reg%s_lr%s_mat%s/' \
                                          % ('1e-5', '0.0001', '3'),
                           'report_filePath':'/hdd/lujunyu/model/meituan/coarse_nn/model/sentiment2/senti_report/',
                            'sr_path':'/hdd/lujunyu/model/meituan/coarse_nn/model/sentiment/ckpt_reg%s_lr%s_mat%s/' \
                                      % ('1e-5', '0.0001', '3'),

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
                self.mt.report('epoch: %d'%i)
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

                    TP_data = self.mt.TP(attr_labels_data, pred_data,mod=dic['test_mod'])
                    FP_data = self.mt.FP(attr_labels_data, pred_data,mod=dic['test_mod'])
                    FN_data = self.mt.FN(attr_labels_data, pred_data,mod=dic['test_mod'])

                    ###Show test message
                    TP_vec.append(TP_data)
                    FP_vec.append(FP_data)
                    FN_vec.append(FN_data)
                    loss_vec.append(test_loss)

                TP_vec = np.concatenate(TP_vec, axis=0)
                FP_vec = np.concatenate(FP_vec, axis=0)
                FN_vec = np.concatenate(FN_vec, axis=0)
                self.mt.report('Val_loss:%.10f' % np.mean(loss_vec))

                _precision = self.mt.precision(TP_vec, FP_vec, 'macro')
                _recall = self.mt.recall(TP_vec, FN_vec, 'macro')
                _f1_score = self.mt.f1_score(_precision, _recall, 'macro')
                self.mt.report('Macro F1 score: %.10f\nMacro precision:%.10f\nMacro recall:%.10f'
                               %(_f1_score,np.mean(_precision),np.mean(_recall)))

                # per class metrics
                per_precision = self.mt.per_precision(TP_vec, FP_vec)
                per_recall = self.mt.per_recall(TP_vec, FN_vec)
                per_f1_score = self.mt.per_f1_score(per_precision,per_recall)
                for i in range(per_precision.shape[0]):
                    self.mt.report('%s f1 score: %.10f precision: %.10f recall: %.10f'
                                   %(self.dg.id_to_aspect_dic[i],per_f1_score[i],per_precision[i],per_recall[i]))



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
            self.mt.report('===========attr============')
            dic['train_step'] = model_dic['train_step']['attr']
            dic['loss'] = model_dic['loss']['attr']
            dic['pred'] = model_dic['pred_labels']['attr']
            dic['test_mod'] = 'attr'

            self.__train__(dic, graph, model_dic['gpu_num'])

            # ##########################
            # train senti (optional)   #
            # ##########################
            self.mt.report('===========senti============')
            dic['train_step'] = model_dic['train_step']['senti']
            dic['loss'] = model_dic['loss']['senti']
            dic['pred'] = model_dic['pred_labels']['senti']
            dic['test_mod'] = 'senti'
            self.__train__(model_dic, graph, model_dic['gpu_num'])

            # ##########################
            # train joint              #
            # ##########################
            self.mt.report('===========joint============')
            dic['train_step'] = model_dic['train_step']['joint']
            dic['loss'] = model_dic['loss']['joint']
            dic['pred'] = model_dic['pred_labels']['joint']
            dic['test_mod'] = 'joint'
            self.__train__(model_dic, graph, model_dic['gpu_num'])