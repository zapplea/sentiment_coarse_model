import numpy as np
import sklearn
import pickle

class Metrics:
    @staticmethod
    def measure(true_labels, pred_labels, id2label_dic):
        true_labels = np.reshape(true_labels, newshape=(-1,)).astype('float32')
        pred_labels = np.reshape(pred_labels, newshape=(-1,)).astype('float32')

        # delete other
        ls_pred = []
        ls_true = []
        for i in range(pred_labels.shape[0]):
            if true_labels[i] != 0 and true_labels != 1:
                ls_pred.append(pred_labels[i])
                ls_true.append(true_labels[i])
        pred_labels = np.array(ls_pred, dtype='float32')
        true_labels = np.array(ls_true, dtype='float32')
        id2label_dic.pop(0)
        all_labels_list = list(range(1, len(id2label_dic)))
        # all_labels_list = list(range(len(id2label_dic)))

        f1 = sklearn.metrics.f1_score(true_labels, pred_labels, labels=all_labels_list, average=None)
        f1_dic = {}
        for i in range(len(f1)):
            label = id2label_dic[i]  # when O is deleted, it should be id2label[i+1]
            f1_dic[label] = f1[i]
        precision = sklearn.metrics.precision_score(true_labels, pred_labels, labels=all_labels_list,
                                                    average=None)
        pre_dic = {}
        for i in range(len(precision)):
            label = id2label_dic[i]  # when O is deleted, it should be id2label[i+1]
            pre_dic[label] = precision[i]

        recall = sklearn.metrics.recall_score(true_labels, pred_labels, labels=all_labels_list, average=None)
        recall_dic = {}
        for i in range(len(recall)):
            label = id2label_dic[i]  # when O is deleted, it should be id2label[i+1]
            recall_dic[label] = recall[i]

        f1_micro = sklearn.metrics.f1_score(true_labels, pred_labels, labels=all_labels_list, average='micro')
        precision_micro = sklearn.metrics.precision_score(true_labels, pred_labels, labels=all_labels_list,
                                                          average='micro')
        recall_micro = sklearn.metrics.recall_score(true_labels, pred_labels, labels=all_labels_list,
                                                    average='micro')
        f1_macro = sklearn.metrics.f1_score(true_labels, pred_labels, labels=all_labels_list, average='macro')
        precision_macro = sklearn.metrics.precision_score(true_labels, pred_labels, labels=all_labels_list,
                                                          average='macro')
        recall_macro = sklearn.metrics.recall_score(true_labels, pred_labels, labels=all_labels_list,
                                                    average='macro')

        accuracy = sklearn.metrics.accuracy_score(true_labels,pred_labels)


        # dictionary of measure
        metrics_dic = {}
        metrics_dic['f1_macro'] = f1_macro
        metrics_dic['f1_micro'] = f1_micro
        metrics_dic['precision_macro'] = precision_macro
        metrics_dic['precision_micro'] = precision_micro
        metrics_dic['recall_macro'] = recall_macro
        metrics_dic['recall_micro'] = recall_micro
        metrics_dic['f1<per-type>'] = f1_dic
        metrics_dic['recall<per-type>'] = recall_dic
        metrics_dic['precision<per-type>'] = pre_dic
        metrics_dic['accuracy'] = accuracy
        return metrics_dic