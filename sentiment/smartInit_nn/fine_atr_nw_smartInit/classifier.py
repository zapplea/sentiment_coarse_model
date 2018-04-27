import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
from sentiment.smartInit_nn.smart_init.smart_initiator import SmartInitiator
from sentiment.functions.attribute_function.attribute_function import AttributeFunction
from sentiment.functions.attribute_function.metrics import Metrics

import tensorflow as tf
import numpy as np

class Classifier:
    def __init__(self, nn_config, data_generator):
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
                graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                    graph.get_tensor_by_name('sentence_lstm/rnn/basic_lstm_cell/kernel:0')))
            # Y_att.shape = (batch size, number of attributes+1)
            Y_att = self.af.attribute_labels_input(graph=graph)
            smartInit = SmartInitiator(self.nn_config)
            if not self.nn_config['is_mat']:
                A, o = self.af.attribute_vec(graph)
                A = A - o
            else:
                A, o = smartInit.attribute_mat(smartInit.smart_initiater(graph), graph)
                # A.shape = (batch size, words num, attributes number, attribute dim)
                A = self.af.words_attribute_mat2vec(H, A, graph)
                o = self.af.words_nonattribute_mat2vec(H, o, graph)
                A = A - o
            # mask
            mask = self.af.mask_for_pad_in_score(X_ids,graph)
            # score.shape = (batch size, attributes num, max sentence length)
            score = self.af.score(A, H,mask, graph)
            graph.add_to_collection('score_pre', score)
            # score.shape = (batch size, attributes num)
            score = tf.reduce_max(score, axis=2)
            graph.add_to_collection('score', score)
            loss = self.af.sigmoid_loss(score,Y_att,graph)
            pred = self.af.prediction(score, graph)
            accuracy = self.mt.accuracy(Y_att, pred, graph)
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
            loss = graph.get_collection('atr_loss')[0]

            pred = graph.get_collection('atr_pred')[0]

            smartInit = graph.get_collection('smartInit')[0]
            score = graph.get_collection('score')[0]
            score_pre = graph.get_collection('score_pre')[0]
            max_false_score = graph.get_collection('max_false_score')[0]
            TP = graph.get_collection('TP')[0]
            FN = graph.get_collection('FN')[0]
            FP = graph.get_collection('FP')[0]
            # attribute function
            init = tf.global_variables_initializer()
        table_data = self.dg.table
        smartInit_data = self.dg.smart_init_embedding

        with graph.device('/gpu:1'):
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(graph=graph, config=config) as sess:
                sess.run(init, feed_dict={smartInit: smartInit_data,table: table_data})

                batch_num = int(self.dg.train_data_size / self.nn_config['batch_size'])
                print('Train set size: ', self.dg.train_data_size, 'Test set size:', self.dg.test_data_size)
                for i in range(self.nn_config['epoch']):
                    loss_vec = []
                    pred_vec = []
                    score_vec = []
                    score_pre_vec = []
                    max_false_score_vec = []
                    Y_att_vec  = []
                    TP_vec = []
                    FP_vec = []
                    FN_vec = []
                    for j in range(batch_num):
                        sentences, Y_att_data = self.dg.train_data_generator(j,i)
                        _, train_loss,TP_data, FP_data, FN_data, pred_data, score_data, max_false_score_data, score_pre_data \
                            = sess.run(
                            [train_step, loss, TP,FP,FN,pred, score, max_false_score, score_pre  ],
                            feed_dict={X: sentences, Y_att: Y_att_data})

                        ###Show training message
                        loss_vec.append(train_loss)
                        TP_vec.append(TP_data)
                        FP_vec.append(FP_data)
                        FN_vec.append(FN_data)
                        for n in range(self.nn_config['batch_size']):
                            pred_vec.append(pred_data[n])
                            score_vec.append(score_data[n])
                            score_pre_vec.append(score_pre_data[n])
                            max_false_score_vec.append(max_false_score_data[n])
                            Y_att_vec.append(Y_att_data[n])
                    if i % 1 == 0:
                        check_num = 1
                        print('Epoch:', i, '\nTraining loss:%.10f' % np.mean(loss_vec))

                        _precision = self.mt.precision(TP_vec,FP_vec,'macro')
                        _recall = self.mt.recall(TP_vec,FN_vec,'macro')
                        _f1_score = self.mt.f1_score(_precision,_recall,'macro')
                        print('F1 score for each class:',_f1_score,'\nPrecision for each class:',_precision,'\nRecall for each class:',_recall)
                        print('Macro F1 score:',np.mean(_f1_score) ,' Macro precision:', np.mean(_precision),' Macro recall:', np.mean(_recall) )

                        _precision = self.mt.precision(TP_vec, FP_vec, 'micro')
                        _recall = self.mt.recall(TP_vec, FN_vec, 'micro')
                        _f1_score = self.mt.f1_score(_precision, _recall, 'micro')
                        print('Micro F1 score:', _f1_score, ' Micro precision:', np.mean(_precision), ' Micro recall:', np.mean(_recall))

                        # # np.random.seed(1)
                        # random_display = np.random.randint(0, 1700, check_num)
                        # pred_check = [[list(self.dg.aspect_dic.keys())[c] for c, rr in enumerate(pred_vec[r]) if rr] for
                        #               r in random_display]
                        # sentences_check = [
                        #     [list(self.dg.dictionary.keys())[word] for word in self.dg.train_sentence_ground_truth[r] if word] for r
                        #     in random_display]
                        # Y_att_check = [[list(self.dg.aspect_dic.keys())[c] for c, rr in
                        #                 enumerate(self.dg.train_attribute_ground_truth[r]) if rr] for r in
                        #                random_display]
                        # score_check = [score_vec[r] for r in random_display]
                        # score_pre_check = [score_pre_vec[r] for r in random_display]
                        # max_false_score_check = [max_false_score_vec[r] for r in random_display]
                        # for n in range(check_num):
                        #     print("sentence id: ", random_display[n], "\nsentence:\n", sentences_check[n], "\npred:\n",
                        #           pred_check[n],
                        #           "\nY_att:\n", Y_att_check[n]
                        #           , "\nscore:\n", score_check[n], "\nmax_false_score:\n", max_false_score_check[n])
                        #     for nn in range(len(score_pre_check[n])):
                        #         if list(self.dg.aspect_dic.keys())[nn] in Y_att_check[n]:
                        #             print(list(self.dg.aspect_dic.keys())[nn] + " score:", score_pre_check[n][nn])

                    if i % 1 == 0 :
                        sentences, Y_att_data = self.dg.test_data_generator()
                        valid_size = Y_att_data.shape[0]
                        loss_vec = []
                        pred_vec = []
                        score_vec = []
                        score_pre_vec = []
                        max_false_score_vec = []
                        Y_att_vec = []
                        TP_vec = []
                        FP_vec = []
                        FN_vec = []
                        batch_size = self.nn_config['batch_size']
                        for i in range(valid_size // batch_size):
                            test_loss,  pred_data, score_data, max_false_score_data, score_pre_data,TP_data, FP_data, FN_data  = sess.run([loss, pred, score, max_false_score, score_pre,TP,FP,FN],
                                                                                                                                feed_dict={X: sentences[i * batch_size:i * batch_size + batch_size],
                                                                                                                               Y_att: Y_att_data[i * batch_size:i * batch_size + batch_size]
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
                                max_false_score_vec.append(max_false_score_data[n])
                        print('\nTest loss:%.10f' % np.mean(loss_vec))

                        _precision = self.mt.precision(TP_vec, FP_vec, 'macro')
                        _recall = self.mt.recall(TP_vec, FN_vec, 'macro')
                        _f1_score = self.mt.f1_score(_precision, _recall, 'macro')
                        print('F1 score for each class:', _f1_score, '\nPrecison for each class:', _precision,
                              '\nRecall for each class:', _recall)
                        print('Macro F1 score:', np.mean(_f1_score), ' Macro precision:', np.mean(_precision),
                              ' Macro recall:', np.mean(_recall))

                        _precision = self.mt.precision(TP_vec, FP_vec, 'micro')
                        _recall = self.mt.recall(TP_vec, FN_vec, 'micro')
                        _f1_score = self.mt.f1_score(_precision, _recall, 'micro')
                        print('Micro F1 score:', _f1_score, ' Micro precision:', np.mean(_precision), ' Micro recall:',np.mean(_recall))
                        # # np.random.seed(1)
                        # random_display = np.random.randint(0, 570, check_num)
                        # pred_check = [[list(self.dg.aspect_dic.keys())[c] for c, rr in enumerate(pred_vec[r]) if rr] for
                        #               r in random_display]
                        # sentences_check = [
                        #     [list(self.dg.dictionary.keys())[word] for word in self.dg.test_sentence_ground_truth[r] if
                        #      word] for r
                        #     in random_display]
                        # Y_att_check = [[list(self.dg.aspect_dic.keys())[c] for c, rr in
                        #                 enumerate(self.dg.test_attribute_ground_truth[r]) if rr] for r in
                        #                random_display]
                        # score_check = [score_vec[r] for r in random_display]
                        # score_pre_check = [score_pre_vec[r] for r in random_display]
                        # max_false_score_check = [max_false_score_vec[r] for r in random_display]
                        # for n in range(check_num):
                        #     print("sentence id: ", random_display[n], "\nsentence:\n", sentences_check[n], "\npred:\n",
                        #           pred_check[n],
                        #           "\nY_att:\n", Y_att_check[n]
                        #           , "\nscore:\n", score_check[n], "\nmax_false_score:\n", max_false_score_check[n])
                        #     for nn in range(len(score_pre_check[n])):
                        #         if list(self.dg.aspect_dic.keys())[nn] in Y_att_check[n]:
                        #             print(list(self.dg.aspect_dic.keys())[nn] + " score:", score_pre_check[n][nn])



