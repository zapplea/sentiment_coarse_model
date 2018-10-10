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


class CoarseAtrTrain:
    def __init__(self,config, data_feeder):
        self.train_config ={
                           'epoch': 1000,
                           'keep_prob_lstm': 0.5,
                           'top_k_data': -1,
                           'early_stop_limit': 100,
                           'tfb_filePath':'/datastore/liu121/sentidata2/resultdata/fine_nn/model/ckpt_reg%s_lr%s_mat%s' \
                                          % ('1e-5', '0.0001', '3'),
                           'report_filePath':'/datastore/liu121/sentidata2/resultdata/fine_nn/report/report_reg%s_lr%s_mat%s.info' \
                                             % ('1e-5', '0.0001', '3')

                        }
        for name in ['tfb_filePath', 'report_filePath']:
            path = Path(self.train_config[name])
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        self.train_config.update(config)
        # self.dg is a class
        self.dg = data_feeder
        # self.cl is a class
        self.mt = Metrics()
        self.tfb = Tfb(self.train_config)


    def train(self,classifier):
        graph, saver = classifier()
        with graph.as_default():
            # input
            X = graph.get_collection('X')[0]
            # labels
            Y_att = graph.get_collection('Y_att')[0]
            # train_step
            train_step = graph.get_collection('opt')[0]
            #
            table = graph.get_collection('table')[0]
            #
            loss = graph.get_collection('atr_loss')[0]

            pred = graph.get_collection('atr_pred')[0]

            keep_prob_lstm = graph.get_collection('keep_prob_lstm')[0]

            # tfb
            micro_f1, micro_pre, micro_rec, macro_f1, macro_pre, macro_rec, tfb_loss = self.tfb.metrics_scalar()
            summ = tf.summary.merge_all()
            writer = tf.summary.FileWriter(self.train_config['tfb_filePath'])
            writer.add_graph(graph)

            # attribute function
            init = tf.global_variables_initializer()
        table_data = self.dg.table

        with graph.device('/gpu:0'):
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(graph=graph, config=config) as sess:
                sess.run(init, feed_dict={table: table_data})
                early_stop_count = 0
                best_f1_score = 0
                for i in range(self.train_config['epoch']):
                    dataset = self.dg.data_generator('train')
                    for att_labels_data, sentences_data in dataset:
                        _, train_loss, pred_data, \
                            = sess.run(
                            [train_step, loss, pred],
                            feed_dict={X: sentences_data, Y_att: att_labels_data,
                                       keep_prob_lstm: self.train_config['keep_prob_lstm']})

                    if i % 1 == 0 and i != 0:
                        loss_vec = []
                        pred_vec = []
                        score_vec = []
                        score_pre_vec = []
                        TP_vec = []
                        FP_vec = []
                        FN_vec = []
                        dataset = self.dg.data_generator('val')
                        for att_labels_data, sentences_data in dataset:
                            test_loss, pred_data, = sess.run(
                                [loss, pred],
                                feed_dict={X: sentences_data,
                                           Y_att: att_labels_data,
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

                        TP_vec = np.concatenate(TP_vec,axis=0)
                        FP_vec = np.concatenate(FP_vec,axis=0)
                        FN_vec = np.concatenate(FN_vec,axis=0)
                        print('Val_loss:%.10f' % np.mean(loss_vec))
                        # tfb
                        tfb_loss.load(np.mean(loss_vec))
                        s = sess.run(summ)
                        writer.add_summary(s, i)



                        _precision = self.mt.precision(TP_vec, FP_vec, 'micro')
                        _recall = self.mt.recall(TP_vec, FN_vec, 'micro')
                        _f1_score = self.mt.f1_score(_precision, _recall, 'micro')
                        print('Micro F1 score:', _f1_score, ' Micro precision:', np.mean(_precision),
                              ' Micro recall:', np.mean(_recall))
                        # tfb
                        micro_f1.load(np.mean(_f1_score))
                        micro_pre.load(np.mean(_precision))
                        micro_rec.load(np.mean(_recall))

                        if best_f1_score<_f1_score:
                            early_stop_count+=1
                        else:
                            early_stop_count=0
                            best_f1_score=_f1_score
                        if early_stop_count>self.train_config['early_stop_limit']:
                            break