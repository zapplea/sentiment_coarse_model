import tensorflow as tf
import math
import numpy as np

from aic.functions.metrics import Metrics

class SentiPrediction:
    def __init__(self,config,model_dic, data_feeder):
        self.pred_config ={'initial_file_path':'',
                           'test_file_path':'',}
        self.pred_config.update(config)
        self.dg = data_feeder
        self.mt = Metrics()
        self.model_dic = model_dic

    def _generate_feed_dict(self,graph, gpu_num, data_dict):
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

    def __prediction__(self,dic,graph,gpu_num):
        sess = dic['sess']
        loss = dic['loss']
        pred = dic['pred']

        loss_vec = []
        TP_vec = []
        FP_vec = []
        FN_vec = []
        dataset = self.dg.data_generator('val')
        for attr_labels_data, senti_labels_data, sentences_data in dataset:
            data_dict = {'X_data': sentences_data, 'Y_att_data': attr_labels_data,
                         'Y_senti_data': senti_labels_data, 'keep_prob': 1.0}
            feed_dict = self._generate_feed_dict(graph=graph, gpu_num=gpu_num, data_dict=data_dict)
            test_loss, pred_data = sess.run(
                [loss, pred],
                feed_dict=feed_dict)
            if dic['test_mod'] == 'attr':
                TP_data = self.mt.TP(attr_labels_data, pred_data)
                FP_data = self.mt.FP(attr_labels_data, pred_data)
                FN_data = self.mt.FN(attr_labels_data, pred_data)
            else:
                d0 = senti_labels_data.shape[0]
                d1 = senti_labels_data.shape[1]
                d2 = senti_labels_data.shape[2]
                senti_labels_data = np.reshape(senti_labels_data, newshape=(d0, d1 * d2))
                pred_data = np.reshape(pred_data, newshape=(d0, d1 * d2))
                TP_data = self.mt.TP(senti_labels_data, pred_data)
                FP_data = self.mt.FP(senti_labels_data, pred_data)
                FN_data = self.mt.FN(senti_labels_data, pred_data)
            ###Show test message
            TP_vec.append(TP_data)
            FP_vec.append(FP_data)
            FN_vec.append(FN_data)
            loss_vec.append(test_loss)

        TP_vec = np.concatenate(TP_vec, axis=0)
        FP_vec = np.concatenate(FP_vec, axis=0)
        FN_vec = np.concatenate(FN_vec, axis=0)
        print('Val_loss:%.10f' % np.mean(loss_vec))

        _precision = self.mt.precision(TP_vec, FP_vec, 'macro')
        _recall = self.mt.recall(TP_vec, FN_vec, 'macro')
        _f1_score = self.mt.f1_score(_precision, _recall, 'macro')
        print('Macro F1 score:', _f1_score, ' Macro precision:', np.mean(_precision),
              ' Macro recall:', np.mean(_recall))


    def prediction(self):
        graph = self.model_dic['graph']
        saver = self.model_dic['saver']

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(graph=graph, config=config) as sess:
            model_file = tf.train.latest_checkpoint(self.pred_config['initial_file_path'])
            saver.restore(sess, model_file)

            # dic = {'sess':sess,'X':X, 'Y_att':Y_att,'Y_senti':Y_senti,'keep_prob_lstm':keep_prob_lstm,'saver':saver}
            dic = {'sess': sess, 'saver': self.model_dic['saver']}
            # ##############
            # train attr   #
            # ##############
            dic['train_step'] = self.model_dic['train_step']['attr']
            dic['loss'] = self.model_dic['loss']['attr']
            dic['pred'] = self.model_dic['pred_labels']['attr']
            dic['test_mod'] = 'attr'

            self.__prediction__(dic, graph, self.model_dic['gpu_num'])

            # ##########################
            # train senti (optional)   #
            # ##########################
            dic['train_step'] = self.model_dic['train_step']['senti']
            dic['loss'] = self.model_dic['loss']['senti']
            dic['pred'] = self.model_dic['pred_labels']['senti']
            dic['test_mod'] = 'senti'
            self.__prediction__(self.model_dic, graph, self.model_dic['gpu_num'])

            # ##########################
            # train joint              #
            # ##########################
            dic['train_step'] = self.model_dic['train_step']['joint']
            dic['loss'] = self.model_dic['loss']['joint']
            dic['pred'] = self.model_dic['pred_labels']['joint']
            dic['test_mod'] = 'joint'
            self.__prediction__(self.model_dic, graph, self.model_dic['gpu_num'])
