import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
from sentiment.multifilter_nn.multi_filter.multi_filter import MultiFilter
from sentiment.functions.attribute_function.attribute_function import AttributeFunction
from sentiment.functions.attribute_function.metrics import Metrics

import tensorflow as tf
import numpy as np

class Classifier:
    def __init__(self, nn_config,data_generator):
        self.nn_config = nn_config
        self.dg = data_generator
        self.af = AttributeFunction(nn_config)
        self.mt = Metrics(self.nn_config)

    def classifier(self):
        graph = tf.Graph()
        with graph.as_default():
            X_ids = self.af.sentences_input(graph=graph)
            words_pad_M = self.af.is_word_padding_input(X_ids, graph)
            X = self.af.lookup_table(X_ids, words_pad_M, graph)
            # lstm
            with tf.variable_scope('sentence_lstm'):
                seq_len = self.af.sequence_length(X_ids, graph)
                # H.shape = (batch size, max_time, cell size)
                H = self.af.sentence_lstm(X, seq_len, graph=graph)
                graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(graph.get_tensor_by_name('sentence_lstm/rnn/basic_lstm_cell/kernel:0')))
            # Y_att.shape = (batch size, number of attributes+1)
            Y_att = self.af.attribute_labels_input(graph=graph)
            #
            mask = self.af.mask_for_pad_in_score(X_ids, graph)
            mf = MultiFilter(self.nn_config)
            multi_score=[]
            for filter_size in self.nn_config['filter_size']:
                filter = mf.filter_generator(X_ids, filter_size)
                X = mf.look_up(X=X, filter=filter, filter_size=filter_size)
                # conv_X.shape = (batch size, max sentence length, last dim of conv)
                conv_X = mf.convolution(X=X, filter_size=filter_size, graph=graph)
                if not self.nn_config['is_mat']:
                    A, o = self.af.attribute_vec(graph)
                    A = A - o
                else:
                    A, o = self.af.attribute_mat(graph)
                    # A.shape = (batch size, words num, attributes number, attribute dim)
                    A = self.af.words_attribute_mat2vec(conv_X,A,graph)
                    o = self.af.words_nonattribute_mat2vec(conv_X,o,graph)
                    A = A-o
                # score.shape = (batch size, attributes number, words num,1)
                score = tf.expand_dims(self.af.score(A,conv_X,mask,graph),axis=3)
                multi_score.append(score)
            # multi_score.shape = (filter numbers, batch size, attributes number, words num,1)
            # multi_kernel_score = (batch size, attributes number, words num, filter numbers)
            multi_kernel_score = tf.concat(multi_score,axis=3)
            # score.shape = (batch size, attributes number ,words num)
            score = tf.reduce_max(multi_kernel_score,axis=3)
            graph.add_to_collection('score_pre', score)
            # score.shape = (batch size, attributes num)
            score = tf.reduce_max(score, axis=2)
            graph.add_to_collection('score', score)

            loss = self.af.sigmoid_loss(score, Y_att, graph)
            pred = self.af.prediction(score,graph)
            accuracy = self.mt.accuracy(Y_att,pred,graph)
        with graph.as_default():
            opt = self.af.optimizer(loss, graph=graph)
            saver = tf.train.Saver()
        return graph, saver

    def train(self):
        graph, saver = self.classifier()
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
            f1 = graph.get_collection('f1')[0]
            #
            loss = graph.get_collection('atr_loss')[0]

            pred = graph.get_collection('atr_pred')[0]

            TP = graph.get_collection('TP')[0]
            FP = graph.get_collection('FP')[0]
            FN = graph.get_collection('FN')[0]
            recall = graph.get_collection('recall')[0]
            precision = graph.get_collection('precision')[0]
            score = graph.get_collection('score')[0]

            # attribute function
            init = tf.global_variables_initializer()
        table_data,_ = self.dg.table_generator()
        with graph.device('/gpu:1'):
            with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                sess.run(init, feed_dict={table: table_data})
                batch_num = int(self.dg.train_data_size / self.nn_config['batch_size'])
                print('Train set size: ',self.dg.train_data_size,'Test set size:',self.dg.test_data_size)
                for i in range(self.nn_config['epoch']):
                    loss_vec = []
                    accuracy_vec = []
                    for j in range(batch_num):
                        sentences, Y_att_data = self.dg.train_data_generator(j)
                        _, train_loss, train_f1_score,TP_data,FP_data,FN_data,precision_data,recall_data, pred_data,score_data\
                            = sess.run([train_step, loss, f1,TP,FP,FN,precision,recall,pred,score],
                                                                            feed_dict={X: sentences, Y_att: Y_att_data})
                        loss_vec.append(train_loss)
                        accuracy_vec.append(train_f1_score)
                        # print(train_accuracy)
                        # print(pred_data)
                        # print(score_data)
                        # print(sum(pred_data == Y_att_data))
                        # print('TP',TP_data,'FP',FP_data,'FN',FN_data,'precision',precision_data,'recall',recall_data)
                    print('Epoch:', i, 'Accuracy:%.10f' % np.mean(accuracy_vec), 'Training loss:%.10f' % np.mean(loss_vec))

                    if i % 5 == 0 and i != 0:
                        print('Test.....')
                        sentences, Y_att_data = self.dg.test_data_generator()
                        valid_size = Y_att_data.shape[0]
                        p = 0
                        l = 0
                        count = 0
                        batch_size = self.nn_config['batch_size']
                        for i in range(valid_size // batch_size):
                            count += 1
                            p += sess.run(f1, feed_dict={X: sentences[i * batch_size:i * batch_size + batch_size],
                                                               Y_att: Y_att_data[
                                                                      i * batch_size:i * batch_size + batch_size]})
                            l += sess.run(loss, feed_dict={X: sentences[i * batch_size:i * batch_size + batch_size],
                                                           Y_att: Y_att_data[
                                                                  i * batch_size:i * batch_size + batch_size]})
                        p = p / count
                        l = l / count
                        print('F1 score:%.10f' % p, 'Testing loss:', l)