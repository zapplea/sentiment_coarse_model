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

class SeqnnTrain:
    def __init__(self,nn_config, data_generator):
        self.nn_config = nn_config
        # self.dg is a class
        self.dg = data_generator
        # self.cl is a class
        self.mt = Metrics(self.nn_config)

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

            score = graph.get_collection('score')[0]
            score_pre = graph.get_collection('score_pre')[0]
            TP = graph.get_collection('TP')[0]
            FN = graph.get_collection('FN')[0]
            FP = graph.get_collection('FP')[0]
            keep_prob_lstm = graph.get_collection('keep_prob_lstm')[0]
            # attribute function
            init = tf.global_variables_initializer()
        table_data = self.dg.table
        aspect_list = list(self.dg.aspect_dic.keys())
        vocab = list(self.dg.dictionary.keys())
        print(aspect_list)

        with graph.device('/gpu:1'):
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(graph=graph, config=config) as sess:
                sess.run(init, feed_dict={table: table_data})

                batch_num = int(self.dg.train_data_size / self.nn_config['batch_size'])
                print('Train set size: ', self.dg.train_data_size, 'Test set size:', self.dg.test_data_size)
                for i in range(self.nn_config['epoch']):
                    loss_vec = []
                    pred_vec = []
                    score_vec = []
                    score_pre_vec = []
                    Y_att_vec = []
                    TP_vec = []
                    FP_vec = []
                    FN_vec = []
                    for j in range(batch_num):
                        sentences, Y_att_data = self.dg.train_data_generator(j)
                        _, train_loss, TP_data, FP_data, FN_data, pred_data, score_data, score_pre_data \
                            = sess.run(
                            [train_step, loss, TP, FP, FN, pred, score, score_pre],
                            feed_dict={X: sentences, Y_att: Y_att_data,
                                       keep_prob_lstm: self.nn_config['keep_prob_lstm']})

                        ###Show training message
                        loss_vec.append(train_loss)
                        TP_vec.append(TP_data)
                        FP_vec.append(FP_data)
                        FN_vec.append(FN_data)
                        for n in range(self.nn_config['batch_size']):
                            pred_vec.append(pred_data[n])
                            score_vec.append(score_data[n])
                            score_pre_vec.append(score_pre_data[n])
                            Y_att_vec.append(Y_att_data[n])

                    # print_op.visualization_train(self.dg,vocab,aspect_list,pred_vec,score_vec,score_pre_vec,i,self.mt,
                    #                              loss_vec, TP_vec, FP_vec, FN_vec)



                    if i % 1 == 0 and i != 0:
                        sentences, Y_att_data = self.dg.test_data_generator()
                        valid_size = Y_att_data.shape[0]
                        loss_vec = []
                        pred_vec = []
                        score_vec = []
                        score_pre_vec = []
                        TP_vec = []
                        FP_vec = []
                        FN_vec = []
                        batch_size = self.nn_config['batch_size']
                        for i in range(valid_size // batch_size):
                            test_loss, pred_data, score_data, score_pre_data, TP_data, FP_data, FN_data = sess.run(
                                [loss, pred, score, score_pre, TP, FP, FN],
                                feed_dict={X: sentences[i * batch_size:i * batch_size + batch_size],
                                           Y_att: Y_att_data[i * batch_size:i * batch_size + batch_size],
                                           keep_prob_lstm: 1.0
                                           })
                            ###Show test message
                            TP_vec.append(TP_data)
                            FP_vec.append(FP_data)
                            FN_vec.append(FN_data)
                            loss_vec.append(test_loss)
                            for n in range(self.nn_config['batch_size']):
                                pred_vec.append(pred_data[n])
                                score_vec.append(score_data[n])
                                score_pre_vec.append(score_pre_data[n])
                        print_op.visualization_test(self.dg,vocab,aspect_list,pred_vec,score_vec,score_pre_vec,i,self.mt,
                                                    loss_vec,TP_vec,FP_vec,FN_vec)