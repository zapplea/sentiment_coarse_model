import getpass
import sys
if  getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif  getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif  getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
from sentiment.functions.attribute_function.metrics import Metrics
from sentiment.functions.tfb.tfb_utils import Tfb
import tensorflow as tf
import numpy as np


class TransferTrain:
    def __init__(self,nn_config, data_generator):
        """
        data_generator is fine grained data generator
        :param nn_config: 
        :param data_generator: 
        """
        self.nn_config = nn_config
        # self.dg is a class
        self.dg = data_generator
        # self.cl is a class
        self.mt = Metrics(self.nn_config)
        self.tfb = Tfb(self.nn_config)

    def train(self,fine_cl, init_data):
        graph,saver = fine_cl.classifier()
        with graph.as_default():
            # input
            X = graph.get_collection('X')[0]
            # labels
            Y_att = graph.get_collection('Y_att')[0]
            # lstm
            for v in tf.all_variables():
                if v.name.startswith('sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0'):
                    bilstm_fw_kernel = v
                elif v.name.startswith('sentence_bilstm/bidirectional_rnn/fw/basic_lstm_cell/bias:0'):
                    bilstm_fw_bias = v
                elif v.name.startswith('sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0'):
                    bilstm_bw_kernel = v
                elif v.name.startswith('sentence_bilstm/bidirectional_rnn/bw/basic_lstm_cell/bias:0'):
                    bilstm_bw_bias = v
            # attribute mention vector or matrix
            if not self.nn_config['is_mat']:
                A=graph.get_collection('A_vec')[0]
                O=graph.get_collection('o_vec')[0]
            else:
                A=graph.get_collection('A_mat')[0]
                O=graph.get_collection('o_mat')[0]
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

            # tfb
            micro_f1,micro_pre,micro_rec,macro_f1,macro_pre,macro_rec, tfb_loss=self.tfb.scalar()
            summ = tf.summary.merge_all()
            writer = tf.summary.FileWriter(self.nn_config['tfb_filePath'])
            writer.add_graph(graph)

            init = tf.global_variables_initializer()
        print(self.dg.aspect_dic)

        with graph.device('/gpu:0'):
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(graph=graph, config=config) as sess:
                sess.run(init, feed_dict={table: init_data['init_table']})
                A.load(init_data['init_A'],sess)
                O.load(init_data['init_O'],sess)
                bilstm_fw_kernel.load(init_data['init_bilstm_fw_kernel'],sess)
                bilstm_fw_bias.load(init_data['init_bilstm_fw_bias'],sess)
                bilstm_bw_kernel.load(init_data['init_bilstm_bw_kernel'],sess)
                bilstm_bw_bias.load(init_data['init_bilstm_bw_bias'],sess)

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
                        # print(score_data)
                        loss_vec.append(train_loss)
                        TP_vec.append(TP_data)
                        FP_vec.append(FP_data)
                        FN_vec.append(FN_data)
                        for n in range(self.nn_config['batch_size']):
                            pred_vec.append(pred_data[n])
                            score_vec.append(score_data[n])
                            score_pre_vec.append(score_pre_data[n])
                            Y_att_vec.append(Y_att_data[n])
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
                    #
                    #     # # np.random.seed(1)
                    #     random_display = np.random.randint(0, 1500, check_num)
                    #     pred_check = [[list(self.dg.aspect_dic.keys())[c] for c, rr in enumerate(pred_vec[r]) if rr] for
                    #                   r in random_display]
                    #     sentences_check = [
                    #         [list(self.dg.dictionary.keys())[word] for word in self.dg.train_sentence_ground_truth[r] if word] for r
                    #         in random_display]
                    #     Y_att_check = [[list(self.dg.aspect_dic.keys())[c] for c, rr in
                    #                     enumerate(self.dg.train_attribute_ground_truth[r]) if rr] for r in
                    #                    random_display]
                    #     score_check = [score_vec[r] for r in random_display]
                    #     score_pre_check = [score_pre_vec[r] for r in random_display]
                    #     for n in range(check_num):
                    #         print("sentence id: ", random_display[n], "\nsentence:\n", sentences_check[n], "\npred:\n",
                    #               pred_check[n],
                    #               "\nY_att:\n", Y_att_check[n]
                    #               , "\nscore:\n", score_check[n])
                    #         for nn in range(len(score_pre_check[n])):
                    #             if list(self.dg.aspect_dic.keys())[nn] in set(Y_att_check[n]) | set(pred_check[n]):
                    #                 print(list(self.dg.aspect_dic.keys())[nn] + " score:", score_pre_check[n][nn])

                    if i % 1 == 0 and i != 0:
                        sentences, Y_att_data = self.dg.test_data_generator()
                        valid_size = Y_att_data.shape[0]
                        loss_vec = []
                        pred_vec = []
                        score_vec = []
                        score_pre_vec = []
                        Y_att_vec = []
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
                        print('\nVal loss:%.10f' % np.mean(loss_vec))

                        tfb_loss.load(np.mean(loss_vec))
                        s = sess.run(summ)
                        writer.add_summary(s,i)
                        # saver.save(sess, self.nn_config['sr_path'])

                        _precision = self.mt.precision(TP_vec, FP_vec, 'macro')
                        _recall = self.mt.recall(TP_vec, FN_vec, 'macro')
                        _f1_score = self.mt.f1_score(_precision, _recall, 'macro')
                        # print('F1 score for each class:', _f1_score, '\nPrecision for each class:', _precision,
                        #       '\nRecall for each class:', _recall)
                        print('Macro F1 score:', np.mean(_f1_score), ' Macro precision:', np.mean(_precision),
                              ' Macro recall:', np.mean(_recall))

                        micro_f1.load(np.mean(_f1_score))
                        micro_pre.load(np.mean(_precision))
                        micro_rec.load(np.mean(_recall))

                        _precision = self.mt.precision(TP_vec, FP_vec, 'micro')
                        _recall = self.mt.recall(TP_vec, FN_vec, 'micro')
                        _f1_score = self.mt.f1_score(_precision, _recall, 'micro')
                        print('Micro F1 score:', _f1_score, ' Micro precision:', np.mean(_precision),
                              ' Micro recall:', np.mean(_recall))
                        macro_f1.load(np.mean(_f1_score))
                        macro_pre.load(np.mean(_precision))
                        macro_rec.load(np.mean(_recall))


                        # # np.random.seed(1)
                        # check_num = 1
                        # random_display = np.random.randint(0, 570, check_num)
                        # pred_check = [[c for c, rr in enumerate(pred_vec[r]) if rr] for
                        #               r in random_display]
                        # sentences_check = [
                        #     [list(self.dg.dictionary.keys())[word] for word in self.dg.test_sentence_ground_truth[r] if
                        #      word] for r
                        #     in random_display]
                        # Y_att_check = [[c for c, rr in
                        #                 enumerate(self.dg.test_attribute_ground_truth[r]) if rr] for r in
                        #                random_display]
                        # score_check = [score_vec[r] for r in random_display]
                        # score_pre_check = [score_pre_vec[r] for r in random_display]
                        # for n in range(check_num):
                        #     print("sentence id: ", random_display[n], "\nsentence:\n", sentences_check[n], "\npred:\n",
                        #           pred_check[n],
                        #           "\nY_att:\n", Y_att_check[n]
                        #           , "\nscore:\n", score_check[n])
                        #     for nn in range(len(score_pre_check[n])):
                        #         if nn in set(Y_att_check[n]) | set(pred_check[n]):
                        #             print(list(self.dg.aspect_dic.keys())[nn]+ '*' , nn , " score:", score_pre_check[n][nn])