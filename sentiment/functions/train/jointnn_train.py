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
import numpy as np
from tensorflow.python import debug as tf_debug

class JointTrain:
    def __init__(self,nn_config, data_generator):
        self.nn_config = nn_config
        self.dg = data_generator
        self.mt = Metrics(self.nn_config)

    def train(self,classifier):
        graph, saver = classifier
        with graph.as_default():
            # input
            X = graph.get_collection('X')[0]
            # labels
            Y_att = graph.get_collection('Y_att')[0]
            Y_senti = graph.get_collection('Y_senti')[0]
            # train_step
            senti_opt = graph.get_collection('atr_opt')[0]
            joint_opt = graph.get_collection('joint_opt')[0]
            #
            table = graph.get_collection('table')[0]
            #
            atr_loss = graph.get_collection('atr_loss')[0]
            joint_loss = graph.get_collection('joint_loss')[0]

            pred = graph.get_collection('prediction')[0]

            score = graph.get_collection('pure_senti_score')[0]
            TP = graph.get_collection('TP')[0]
            FN = graph.get_collection('FN')[0]
            FP = graph.get_collection('FP')[0]
            keep_prob_lstm = graph.get_collection('keep_prob_lstm')[0]
            check = graph.get_collection('check')[0]
            # attribute function
            init = tf.global_variables_initializer()
        max_f1_score = 0
        table_data = self.dg.table
        print(self.dg.aspect_dic)

        with graph.device('/gpu:1'):
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(graph=graph, config=config) as sess:
                sess.run(init, feed_dict={table: table_data})
                # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

                batch_num = int(self.dg.train_data_size / self.nn_config['batch_size'])
                print('Train set size: ', self.dg.train_data_size, 'validation set size:', self.dg.val_data_size)
                for i in range(self.nn_config['epoch']):
                    loss_vec = []
                    TP_vec = []
                    FP_vec = []
                    FN_vec = []
                    for j in range(batch_num):
                        sentences, Y_att_data ,Y_senti_data= self.dg.data_generator(j,'train')
                        _, joint_train_loss, TP_data, FP_data, FN_data, pred_data, score_data ,check_data\
                            = sess.run(
                            [joint_opt, joint_loss, TP, FP, FN, pred, score ,check],
                            feed_dict={X: sentences, Y_att: Y_att_data,Y_senti:Y_senti_data,
                                       keep_prob_lstm: self.nn_config['keep_prob_lstm']})

                        ###Show training message
                        # print(j,train_loss)
                        # print(check_data)
                        # print('Batch :',j,'Training loss:%0.8f'%train_loss)
                        # #
                        # random_display = np.random.randint(0, self.nn_config['batch_size'])
                        # pred_check = [(list(self.dg.aspect_dic.keys())[rrr],list(self.dg.senti_dic.keys())[c]) for rrr in range(self.nn_config['attributes_num']) for c, rr in enumerate(pred_data[random_display][rrr]) if rr]
                        # Y_att_check = [(list(self.dg.aspect_dic.keys())[rrr],list(self.dg.senti_dic.keys())[c]) for rrr in range(self.nn_config['attributes_num']) for c, rr in enumerate(Y_senti_data[random_display][rrr]) if rr]
                        # sentences_check = [list(self.dg.dictionary.keys())[word] for word in sentences[random_display] if word]
                        # senti_score_check = word_score_data[random_display]
                        # print("sentence id: ", random_display, "\nsentence:\n", sentences_check,"\nreview length:\n", len(sentences_check), "\npred:\n",pred_check,"\nY_att:\n", Y_att_check,'\nsentiment score:',word_score_data)
                        #
                        loss_vec.append(train_loss)
                        TP_vec.append(TP_data)
                        FP_vec.append(FP_data)
                        FN_vec.append(FN_data)

                    print('Epoch:', i, '\nTraining loss:%.10f' % np.mean(loss_vec))
                    if i % 1 == 0:
                        _precision = self.mt.precision(TP_vec,FP_vec,'macro')
                        _recall = self.mt.recall(TP_vec,FN_vec,'macro')
                        _f1_score = self.mt.f1_score(_precision,_recall,'macro')
                        print('F1 score for each class:',_f1_score,'\nPrecision for each class:',_precision,'\nRecall for each class:',_recall)
                        print('Macro F1 score:',np.mean(_f1_score) ,' Macro precision:', np.mean(_precision),' Macro recall:', np.mean(_recall) )

                        _precision = self.mt.precision(TP_vec, FP_vec, 'micro')
                        _recall = self.mt.recall(TP_vec, FN_vec, 'micro')
                        _f1_score = self.mt.f1_score(_precision, _recall, 'micro')
                        print('Micro F1 score:', _f1_score, ' Micro precision:', np.mean(_precision), ' Micro recall:', np.mean(_recall))


                    if i % 1 == 0 and i != 0:
                        loss_vec = []
                        pred_vec = []
                        score_vec = []
                        score_pre_vec = []
                        Y_att_vec = []
                        TP_vec = []
                        FP_vec = []
                        FN_vec = []
                        val_batch_num = int(self.dg.val_data_size / self.nn_config['batch_size'])
                        for j in range(val_batch_num):
                            sentences, Y_att_data,Y_senti_data = self.dg.data_generator(j,'val')
                            val_loss, pred_data, score_data, TP_data, FP_data, FN_data = sess.run(
                                [loss, pred, score, TP, FP, FN],
                                feed_dict={X: sentences,
                                           Y_att: Y_att_data,
                                           Y_senti: Y_senti_data,
                                           keep_prob_lstm: 1.0
                                           })
                            ##Show test message
                            random_display = np.random.randint(0, self.nn_config['batch_size'])
                            # if random_display % 6 == 0:
                            #     display_start = random_display * self.nn_config['max_review_length']
                            #     display_end = (random_display+1) * self.nn_config['max_review_length']
                            #     pred_check = [list(self.dg.aspect_dic.keys())[c] for c, rr in enumerate(np.sum(pred_data[display_start:display_end],axis=0)) if rr]
                            #     Y_att_check = [list(self.dg.aspect_dic.keys())[c] for c, rr in enumerate(np.sum(true_labels_data[display_start:display_end],axis=0)) if rr]
                            #     sentences_check = [[list(self.dg.dictionary.keys())[word] for word in s if word != self.nn_config['padding_word_index']] for s in sentences[random_display] if [list(self.dg.dictionary.keys())[word] for word in s if word != self.nn_config['padding_word_index']]]
                            #     coarse_atr_score_check = score_data[display_start:display_end][range(len(sentences_check))]
                            #     print('\nBatch:',j,"\nsentence id: ", random_display, "\nsentence:\n", sentences_check,"\nreview length:\n", len(sentences_check), "\npred:\n",pred_check,"\nY_att:\n", Y_att_check,'\ncoarse score:',coarse_atr_score_check)
                            TP_vec.append(TP_data)
                            FP_vec.append(FP_data)
                            FN_vec.append(FN_data)
                            loss_vec.append(val_loss)
                            for n in range(self.nn_config['batch_size']):
                                pred_vec.append(pred_data[n])
                                score_vec.append(score_data[n])
                        print('\nVal_loss:%.10f' % np.mean(loss_vec))

                        _precision = self.mt.precision(TP_vec, FP_vec, 'macro')
                        _recall = self.mt.recall(TP_vec, FN_vec, 'macro')
                        _f1_score = self.mt.f1_score(_precision, _recall, 'macro')
                        print('F1 score for each class:', _f1_score, '\nPrecision for each class:', _precision,
                              '\nRecall for each class:', _recall)
                        print('Macro F1 score:', np.mean(_f1_score), ' Macro precision:', np.mean(_precision),
                              ' Macro recall:', np.mean(_recall))

                        _precision = self.mt.precision(TP_vec, FP_vec, 'micro')
                        _recall = self.mt.recall(TP_vec, FN_vec, 'micro')
                        _f1_score = self.mt.f1_score(_precision, _recall, 'micro')
                        print('Micro F1 score:', _f1_score, ' Micro precision:', np.mean(_precision),
                              ' Micro recall:', np.mean(_recall))

                        if max_f1_score < _f1_score:
                            max_f1_score = _f1_score
                            # saver.save(sess,self.nn_config['model_save_path'],global_step=i+1)
                        print('Max Micro F1 score: ',max_f1_score)
