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


class CoarseSentiTrain:
    def __init__(self,config, data_feeder):
        self.train_config ={
                           'epoch': 100,
                           'keep_prob_lstm': 0.5,
                           'top_k_data': -1,
                           'early_stop_limit': 2,
                           'tfb_filePath':'/hdd/lujunyu/model/meituan/coarse_nn/model/sentiment2/ckpt_reg%s_lr%s_mat%s/' \
                                          % ('1e-5', '0.0001', '3'),
                           'init_model':'/hdd/lujunyu/model/meituan/coarse_nn/model/sentiment/ckpt_reg%s_lr%s_mat%s/' \
                                          % ('1e-5', '0.0001', '3'),
                           'report_filePath':'/hdd/lujunyu/model/meituan/coarse_nn/model/sentiment2/senti_report/',
                            'sr_path':'/hdd/lujunyu/model/meituan/coarse_nn/model/sentiment/ckpt_reg%s_lr%s_mat%s/' \
                                      % ('1e-5', '0.0001', '3'),

                        }
        for name in ['tfb_filePath', 'report_filePath','sr_path']:
            path = Path(self.train_config[name])
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        self.train_config['report_filePath'] = self.train_config['report_filePath'] +'report_reg%s_lr%s_mat%s.info'% ('1e-5', '0.0001', '3')
        self.train_config.update(config)
        # self.dg is a class
        self.dg = data_feeder
        # self.cl is a class
        self.mt = Metrics()
        self.tfb = Tfb(self.train_config)

    def __train__(self,dic):
        sess = dic['sess']
        train_step = dic['train_step']
        loss = dic['loss']
        pred = dic['pred']
        X=dic['X']
        Y_att = dic['Y_att']
        Y_senti = dic['Y_senti']
        keep_prob_lstm = dic['keep_prob_lstm']
        saver = dic['saver']
        early_stop_count = 0
        best_f1_score = 0

        for i in range(self.train_config['epoch']):

            dataset = self.dg.data_generator('train')
            for attr_labels_data, senti_labels_data, sentences_data in dataset:
                _, train_loss, pred_data, \
                    = sess.run(
                    [train_step, loss, pred, ],
                    feed_dict={X: sentences_data, Y_att: attr_labels_data, Y_senti: senti_labels_data,
                               keep_prob_lstm: self.train_config['keep_prob_lstm']})

            if i % 1 == 0 and i != 0:
                loss_vec = []
                TP_vec = []
                FP_vec = []
                FN_vec = []
                dataset = self.dg.data_generator('val')
                for att_labels_data, senti_labels_data, sentences_data in dataset:
                    test_loss, pred_data = sess.run(
                        [loss, pred],
                        feed_dict={X: sentences_data,
                                   Y_att: att_labels_data, Y_senti: senti_labels_data,
                                   keep_prob_lstm: 1.0
                                   })
                    TP_data = self.mt.TP(att_labels_data, pred_data)
                    FP_data = self.mt.FP(att_labels_data, pred_data)
                    FN_data = self.mt.FN(att_labels_data, pred_data)
                    ###Show test message
                    TP_vec.append(TP_data)
                    FP_vec.append(FP_data)
                    FN_vec.append(FN_data)
                    loss_vec.append(test_loss)

                TP_vec = np.concatenate(TP_vec, axis=0)
                FP_vec = np.concatenate(FP_vec, axis=0)
                FN_vec = np.concatenate(FN_vec, axis=0)
                print('Val_loss:%.10f' % np.mean(loss_vec))

                _precision = self.mt.precision(TP_vec, FP_vec, 'micro')
                _recall = self.mt.recall(TP_vec, FN_vec, 'micro')
                _f1_score = self.mt.f1_score(_precision, _recall, 'micro')
                print('Micro F1 score:', _f1_score, ' Micro precision:', np.mean(_precision),
                      ' Micro recall:', np.mean(_recall))

                if best_f1_score > _f1_score:
                    early_stop_count += 1
                else:
                    early_stop_count = 0
                    best_f1_score = _f1_score
                    _save_path = saver.save(sess, self.train_config['sr_path'])
                    print("succ saving model in " + _save_path)
                if early_stop_count > self.train_config['early_stop_limit']:
                    break

    def train(self,classifier):
        graph, saver = classifier()
        with graph.as_default():
            # input
            X = graph.get_collection('X')[0]
            # labels
            Y_att = graph.get_collection('Y_att')[0]
            Y_senti = graph.get_collection('Y_senti')[0]
            # train_step
            attr_train_step = graph.get_collection('attr_opt')[0]
            senti_train_step = graph.get_collection('senti_opt')[0]
            joint_train_step = graph.get_collection('joint_opt')[0]
            #
            table = graph.get_collection('table')[0]
            #loss
            attr_loss = graph.get_collection('atr_loss')[0]
            senti_loss = graph.get_collection('senti_loss')[0]
            joint_loss = graph.get_collection('joint_loss')[0]

            # pred
            attr_pred = graph.get_collection('atr_pred')[0]
            senti_pred = graph.get_collection('senti_pred')[0]
            joint_pred = graph.get_collection('joint_pred')[0]

            keep_prob_lstm = graph.get_collection('keep_prob_lstm')[0]

            # attribute function
            init = tf.global_variables_initializer()
        table_data = self.dg.table

        with graph.device('/gpu:0'):
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(graph=graph, config=config) as sess:
                sess.run(init, feed_dict={table: table_data})
                if self.train_config['init_model']:
                    model_path = tf.train.latest_checkpoint(self.train_config['init_model'])
                    saver.restore(sess, model_path)
                    print("sucess init %s" % self.train_config['init_model'])
                dic = {'sess':sess,'X':X, 'Y_att':Y_att,'Y_senti':Y_senti,'keep_prob_lstm':keep_prob_lstm,'saver':saver}
                # ##############
                # train attr   #
                # ##############
                # dic['train_step'] = attr_train_step
                # dic['loss'] = attr_loss
                # dic['pred'] = attr_pred
                # self.__train__(dic)

                # ##########################
                # train senti (optional)   #
                # ##########################
                # dic['train_step'] = senti_train_step
                # dic['loss'] = senti_loss
                # dic['pred'] = senti_pred
                # self.__train__(dic)

                # ##########################
                # train joint              #
                # ##########################
                dic['train_step'] = joint_train_step
                dic['loss'] = joint_loss
                dic['pred'] = joint_pred
                self.__train__(dic)