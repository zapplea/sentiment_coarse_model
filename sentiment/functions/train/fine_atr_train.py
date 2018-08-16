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
from sentiment.functions.attribute_function.metrics import Metrics
import tensorflow as tf
import numpy as np
import sentiment.functions.train.fine_print_op as print_op
from sentiment.functions.tfb.tfb_utils import Tfb

class FineAtrTrain:
    def __init__(self,nn_config, data_feeder):
        self.nn_config = nn_config
        # self.dg is a class
        self.dg = data_feeder
        # self.cl is a class
        self.mt = Metrics(self.nn_config)
        self.tfb = Tfb(self.nn_config)


    def train(self,classifier):
        graph, saver = classifier
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
            # Attribute mat
            A_mat = graph.get_collection('A_mat')[0]

            score = graph.get_collection('score')[0]
            score_pre = graph.get_collection('score_pre')[0]
            TP = graph.get_collection('TP')[0]
            FN = graph.get_collection('FN')[0]
            FP = graph.get_collection('FP')[0]
            keep_prob_lstm = graph.get_collection('keep_prob_lstm')[0]

            # tfb
            micro_f1, micro_pre, micro_rec, macro_f1, macro_pre, macro_rec, tfb_loss = self.tfb.metrics_scalar()
            summ = tf.summary.merge_all()
            writer = tf.summary.FileWriter(self.nn_config['tfb_filePath'])
            writer.add_graph(graph)

            # attribute function
            init = tf.global_variables_initializer()
        table_data = self.dg.table
        aspect_list = list(self.dg.aspect_dic.keys())
        vocab = list(self.dg.dictionary.keys())
        print(aspect_list)

        with graph.device('/gpu:0'):
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(graph=graph, config=config) as sess:
                sess.run(init, feed_dict={table: table_data})
                early_stop_count = 0
                for i in range(self.nn_config['epoch']):
                    print('\nepoch: ',i)
                    loss_vec = []
                    pred_vec = []
                    score_vec = []
                    score_pre_vec = []
                    Y_att_vec = []
                    TP_vec = []
                    FP_vec = []
                    FN_vec = []
                    aspect_data = sess.run(A_mat)
                    print('aspect_data0: ')
                    print(aspect_data[0])
                    dataset = self.dg.data_generator('train')
                    for att_labels_data, sentences_data in dataset:
                        _, train_loss, TP_data, FP_data, FN_data, pred_data, score_data, score_pre_data \
                            = sess.run(
                            [train_step, loss, TP, FP, FN, pred, score, score_pre],
                            feed_dict={X: sentences_data, Y_att: att_labels_data,
                                       keep_prob_lstm: self.nn_config['keep_prob_lstm']})

                        ###Show training message
                        # loss_vec.append(train_loss)
                        # TP_vec.append(TP_data)
                        # FP_vec.append(FP_data)
                        # FN_vec.append(FN_data)
                        # for n in range(self.nn_config['batch_size']):
                        #     pred_vec.append(pred_data[n])
                        #     score_vec.append(score_data[n])
                        #     score_pre_vec.append(score_pre_data[n])
                        #     Y_att_vec.append(Y_att_data[n])

                    # print_op.visualization_train(self.dg,vocab,aspect_list,pred_vec,score_vec,score_pre_vec,i,self.mt,
                    #                              loss_vec, TP_vec, FP_vec, FN_vec)



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
                            test_loss, pred_data, score_data, score_pre_data, TP_data, FP_data, FN_data = sess.run(
                                [loss, pred, score, score_pre, TP, FP, FN],
                                feed_dict={X: sentences_data,
                                           Y_att: att_labels_data,
                                           keep_prob_lstm: 1.0
                                           })
                            ###Show test message
                            TP_vec.append(TP_data)
                            FP_vec.append(FP_data)
                            FN_vec.append(FN_data)
                            loss_vec.append(test_loss)
                            # for n in range(self.nn_config['batch_size']):
                            #     pred_vec.append(pred_data[n])
                            #     score_vec.append(score_data[n])
                            #     score_pre_vec.append(score_pre_data[n])
                        # print_op.visualization_test(self.dg,vocab,aspect_list,pred_vec,score_vec,score_pre_vec,i,self.mt,
                        #                             loss_vec,TP_vec,FP_vec,FN_vec)
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