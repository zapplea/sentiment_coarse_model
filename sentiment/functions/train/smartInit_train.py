import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
from sentiment.functions.attribute_function.metrics import Metrics
import tensorflow as tf

class SmartInitTrain:
    def __init__(self,nn_config, data_generator):
        self.nn_config = nn_config
        # self.dg is a class
        self.dg = data_generator
        # self.cl is a class
        self.mt = Metrics(self.nn_config)

    def train(self,classifier):
        graph, saver = classifier()
        with graph.as_default():
            # input
            X = graph.get_collection('X')[0]
            # labels
            Y_att = graph.get_collection('Y_att')[0]
            # name_list
            name_list_vec = graph.get_collection('name_list_vec')[0]
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
                    max_false_score_vec = []
                    Y_att_vec  = []
                    TP_vec = []
                    FP_vec = []
                    FN_vec = []
                    for j in range(batch_num):
                        sentences, Y_att_data , name_list_data = self.dg.train_data_generator(j,i)
                        _, train_loss,TP_data, FP_data, FN_data, pred_data, score_data, score_pre_data \
                            = sess.run(
                            [train_step, loss, TP,FP,FN,pred, score, score_pre  ],
                            feed_dict={X: sentences, Y_att: Y_att_data,name_list_vec: name_list_data,keep_prob_lstm:0.5})

                    #     ###Show training message
                    #     loss_vec.append(train_loss)
                    #     TP_vec.append(TP_data)
                    #     FP_vec.append(FP_data)
                    #     FN_vec.append(FN_data)
                    #     for n in range(self.nn_config['batch_size']):
                    #         pred_vec.append(pred_data[n])
                    #         score_vec.append(score_data[n])
                    #         score_pre_vec.append(score_pre_data[n])
                    #         Y_att_vec.append(Y_att_data[n])
                    # if i % 1 == 0:
                    #     check_num = 1
                    #     print('Epoch:', i, '\nTraining loss:%.10f' % np.mean(loss_vec))
                    #
                    #     _precision = self.mt.precision(TP_vec,FP_vec,'macro')
                    #     _recall = self.mt.recall(TP_vec,FN_vec,'macro')
                    #     _f1_score = self.mt.f1_score(_precision,_recall,'macro')
                    #     print('F1 score for each class:',_f1_score,'\nPrecision for each class:',_precision,'\nRecall for each class:',_recall)
                    #     print('Macro F1 score:',np.mean(_f1_score) ,' Macro precision:', np.mean(_precision),' Macro recall:', np.mean(_recall) )
                    #
                    #     _precision = self.mt.precision(TP_vec, FP_vec, 'micro')
                    #     _recall = self.mt.recall(TP_vec, FN_vec, 'micro')
                    #     _f1_score = self.mt.f1_score(_precision, _recall, 'micro')
                    #     print('Micro F1 score:', _f1_score, ' Micro precision:', np.mean(_precision), ' Micro recall:', np.mean(_recall))

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
                        sentences, Y_att_data , name_list_data = self.dg.test_data_generator()
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
                            test_loss,  pred_data, score_data, score_pre_data,TP_data, FP_data, FN_data  = sess.run([loss, pred, score, score_pre,TP,FP,FN],
                                                                                                                                feed_dict={X: sentences[i * batch_size:i * batch_size + batch_size],
                                                                                                                               Y_att: Y_att_data[i * batch_size:i * batch_size + batch_size],
                                                                                                                               name_list_vec:name_list_data[i * batch_size:i * batch_size + batch_size],
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