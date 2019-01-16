import tensorflow as tf
import math
import numpy as np

from aic.functions.metrics import Metrics

class SentiPrediction:
    def __init__(self,config, data_feeder):
        self.pred_config ={'initial_filePath':'',
                           'report_filePath':'',}
        self.pred_config.update(config)
        self.dg = data_feeder

    def generate_feed_dict(self,graph, gpu_num, data_dict):
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

    def __prediction__(self,dic, graph, gpu_num):
        sess = dic['sess']
        attr_pred = dic['pred']['attr']
        senti_pred = dic['pred']['senti']
        dataset = self.dg.data_generator('val')
        pred_data = []
        label_data = []
        for attr_labels_data, senti_labels_data, sentences_data in dataset:
            data_dict = {'X_data': sentences_data, 'Y_att_data': attr_labels_data,
                         'Y_senti_data': senti_labels_data, 'keep_prob': 1.0}
            feed_dict = self.generate_feed_dict(graph=graph, gpu_num=gpu_num, data_dict=data_dict)
            attr_pred_data, senti_pred_data = sess.run([attr_pred, senti_pred],feed_dict=feed_dict)
            if dic['pred_mod'] == 'attr':
                pred_data.append(attr_pred_data)
                label_data.append(attr_labels_data)
            else:
                pred_data.append(senti_pred_data)
                label_data.append(senti_labels_data)
        pred_data = np.concatenate(pred_data,axis=0)
        label_data = np.concatenate(label_data,axis=0)

    def prediction(self,model_dic):
        graph = model_dic['graph']
        with graph.as_default():
            # shape = (batch size*max review length, attributes num, words num)
            attr_score_list = tf.get_collection('attr_score')
            # shape =


        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(graph=graph, config=config) as sess:
            print('initial path: %s' % self.pred_config['initial_path'])
            model_file = tf.train.latest_checkpoint(self.pred_config['initial_path'])
            model_dic['saver'].restore(sess, model_file)
            print('restore successful')

            dic = {'sess': sess,}
            # ##############
            # pred attr    #
            # ##############
            dic['pred'] = {'attr': model_dic['pred_labels']['attr'], 'senti': model_dic['pred_labels']['joint']}
            dic['pred_mod'] = 'attr'
            self.__prediction__(dic, graph, model_dic['gpu_num'])
            # ##############
            # pred senti   #
            # ##############
            dic['pred'] = {'attr': model_dic['pred_labels']['attr'], 'senti': model_dic['pred_labels']['senti']}
            # dic['pred'] = {'attr': model_dic['pred_labels']['attr'], 'senti': model_dic['pred_labels']['joint']}
            dic['pred_mod'] = 'senti'
            self.__prediction__(dic, graph, model_dic['gpu_num'])
