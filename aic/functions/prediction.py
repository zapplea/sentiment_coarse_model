import tensorflow as tf
import math
import numpy as np
from pathlib import Path

from aic.functions.metrics import Metrics

class SentiPrediction:
    def __init__(self,config, data_feeder):
        self.pred_config ={'initial_filePath':'',
                           'report_filePath':'',
                           'padding_word_index': 116140,
                           'attributes_num':20}
        self.pred_config.update(config)
        dir_ls = ['report_filePath',]
        for name in dir_ls:
            path = Path(self.pred_config[name])
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        self.pred_config['report_filePath']=self.pred_config['report_filePath']+'report_reg%s_lr%s_mat%s.info'% \
                                               (str(self.pred_config['reg_rate']), str(self.pred_config['lr']), str(self.pred_config['attribute_mat_size']))
        self.dg = data_feeder
        train_dataset = self.dg('train')
        self.aspect_dic = train_dataset.aspect_dic
        # TODO: check the dictionary is id to word or word to id
        self.dictionary = train_dataset.dictionary
        self.outf = open(self.pred_config['report_filePath'],'w+')

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

    def TP(self,true_labels,pred_labels):
        result = true_labels*pred_labels
        return np.count_nonzero(result,axis=0).astype('float32')

    def TN(self,true_labels,pred_labels):
        result = (pred_labels-1)*(true_labels-1)
        return np.count_nonzero(result,axis=0).astype('float32')

    def FP(self,true_labels, pred_labels):
        result = pred_labels*(true_labels-1)
        return np.count_nonzero(result, axis=0).astype('float32')

    def FN(self,true_labels, pred_labels):
        result = (pred_labels-1)*true_labels
        return np.count_nonzero(result,axis=0).astype('float32')

    def __prediction__(self,dic, graph, gpu_num):
        sess = dic['sess']
        attr_pred = dic['pred']['attr']
        senti_pred = dic['pred']['senti']
        dataset = self.dg.data_generator('val')
        attr_pred_value_ls = []
        attr_label_value_ls = []
        senti_pred_value_ls = []
        senti_label_value_ls = []
        item1_value_ls = []
        item2_value_ls = []
        attr_score_value_ls = []
        sent_value_ls = []
        for attr_labels_data, senti_labels_data, sentences_data in dataset:
            sent_value_ls.append(sentences_data)
            data_dict = {'X_data': sentences_data, 'Y_att_data': attr_labels_data,
                         'Y_senti_data': senti_labels_data, 'keep_prob': 1.0}
            feed_dict = self.generate_feed_dict(graph=graph, gpu_num=gpu_num, data_dict=data_dict)
            attr_pred_value, senti_pred_value = sess.run([attr_pred, senti_pred],feed_dict=feed_dict)

            ## attribute ##
            attr_pred_value_ls.append(attr_pred_value)
            attr_label_value_ls.append(attr_labels_data)
            # (batch size*19, attributes num, words num)
            attr_score_value = np.concatenate(sess.run(tf.get_collection('attr_score'),feed_dict=feed_dict),axis=0)
            shape = attr_score_value.shape
            # (batch size, 19, attributes num, words num)
            attr_score_value = np.reshape(attr_score_value,newshape=(shape[0]/19,19,shape[1],shape[2]))
            attr_score_value_ls.append(attr_score_value)

            ## sentiment ##
            senti_pred_value_ls.append(senti_pred_value)
            senti_label_value_ls.append(senti_labels_data)

            # (batch size, number of words, 3+3*attributes number)
            item1_value = np.concatenate(sess.run(tf.get_collection('item1'), feed_dict=feed_dict), axis=0)
            # (batch size, 3+3*attributes number, number of words)
            item1_value = np.transpose(item1_value, [0, 2, 1])
            shape = item1_value.shape
            # (batch size, 1+attributes number, 3, number of words)
            item1_value = np.reshape(item1_value, newshape=(shape[0], shape[1] / 3, 3, shape[2]))
            shape = item1_value.shape
            # (batch size, 1+attributes number, 3, number of words)
            item1_value = np.reshape(item1_value,newshape=(shape[0]/19,19,shape[1],shape[2],shape[3]))
            # (batch size, number of attributes+1, number of words)
            item2_value = np.concatenate(sess.run(tf.get_collection('item2'), feed_dict=feed_dict), axis=0)
            shape = item2_value.shape
            item2_value = np.reshape(item2_value,newshape=(shape[0]/19,19,shape[1],shape[2]))

            item1_value_ls.append(item1_value)
            item2_value_ls.append(item2_value)


        attr_pred_data = np.concatenate(attr_pred_value_ls,axis=0)
        attr_label_data = np.concatenate(attr_label_value_ls,axis=0)
        sent_data = np.concatenate(sent_value_ls,axis=0)
        attr_score_data = np.concatenate(attr_score_value_ls,axis=0)

        senti_pred_data = np.concatenate(senti_pred_value_ls, axis=0)
        senti_label_data = np.concatenate(senti_label_value_ls, axis=0)
        item1_data = np.concatenate(item1_value_ls,axis=0)
        item2_data = np.concatenate(item2_value_ls,axis=0)

        return attr_pred_data,attr_label_data, attr_score_data, senti_pred_data, senti_label_data,item1_data,item2_data, sent_data

    def metrics(self,true_label,pred_label):
        tp = self.TP(true_label, pred_label)
        fp = self.FP(true_label, pred_label)
        fn = self.FN(true_label, pred_label)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2*precision*recall/(precision+recall+1e-10)
        return f1

    def report(self,info,):
        self.outf.write(info)

    def review_len(self,review):
        condition = np.equal(review,self.pred_config['padding_word_index'])
        condition = np.greater(np.sum(np.where(condition,np.zeros_like(review).astype('int32'),np.ones_like(review).astype('int32')),axis=1),0)
        return np.sum(np.where(condition,np.ones_like(condition).astype('int32'),np.zeros_like(condition).astype('int32'))).astype('int32')

    def sent_translate(self,review,):

        # Done:TODO: elimiate padded sentence
        review_length = self.review_len(review)
        review_txt = []
        for i in range(review_length):
            sentence = review[i]
            sentence_txt = []
            for word_id in sentence:
                if word_id == self.pred_config['padding_word_index']:
                    break
                word = self.dictionary[word_id]
                sentence_txt.append(word)
            review_txt.append(sentence_txt)
        return review_txt

    def aspect_translate(self,label):
        label_txt = []
        for i in range(label.shape[0]):
            if label[i] == 1:
                label_txt.append(self.aspect_dic[i])
        return label_txt

    def prediction(self,model_dic):
        graph = model_dic['graph']
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(graph=graph, config=config) as sess:
            print('initial path: %s' % self.pred_config['initial_path'])
            model_file = tf.train.latest_checkpoint(self.pred_config['initial_path'])
            model_dic['saver'].restore(sess, model_file)
            print('restore successful')

            dic = {'sess': sess,}
            # ##############
            # joint    #
            # ##############
            dic['pred'] = {'attr': model_dic['pred_labels']['attr'], 'senti': model_dic['pred_labels']['joint']}
            attr_pred_data, attr_label_data, attr_score_data, senti_pred_data, senti_label_data, item1_data, item2_data, sent_data = self.__prediction__(dic, graph, model_dic['gpu_num'])

            shape = attr_pred_data.shape
            for i in range(shape[0]):
                pred_label = attr_pred_data[i]
                true_label = attr_label_data[i]
                f1 = self.metrics(true_label,pred_label)
                if f1>0.7:
                    # shape=(19, word num)
                    sentence_txt = self.sent_translate(sent_data[i])
                    # (19, attributes num, words num)
                    # DONE:TODO: need to eliminate influence of paded word in a sentence for attr_score
                    attr_score = attr_score_data[i]
                    # attr percent is a list
                    pred_label_txt = self.aspect_translate(attr_pred_data[i])
                    true_label_txt = self.aspect_translate(attr_label_data[i])
                    self.report('pred label: %s\n'%(' ,'.join(pred_label_txt)))
                    self.report('true label: %s\n'%(' ,'.join(true_label_txt)))
                    for j in range(len(sentence_txt)):
                        self.report('sentence_%d: %s\n'%(j,sentence_txt[j]))
                        length = len(sentence_txt[j])
                        for l in range(self.pred_config['attributes_num']):
                            self.report('aspect: %s\n'%self.aspect_dic[l])
                            self.report('attr_percent: %s\n'%str(attr_score[j,l,:length]))
            # ##############
            # senti        #
            # ##############
            dic['pred'] = {'attr': model_dic['pred_labels']['attr'], 'senti': model_dic['pred_labels']['senti']}
            # dic['pred'] = {'attr': model_dic['pred_labels']['attr'], 'senti': model_dic['pred_labels']['joint']}
