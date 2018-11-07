import tensorflow as tf
import numpy as np

class Metrics:
    def TP(self,true_labels,pred_labels,mod = 'attr'):
        if mod =='attr':
            result = true_labels*pred_labels
        else:
            result = np.sum(true_labels*pred_labels,axis=2)
        return np.count_nonzero(result,axis=0).astype('float32')

    def TN(self,true_labels,pred_labels, mod='attr'):
        if mod == 'attr':
            result = (pred_labels-1)*(true_labels-1)
        else:
            result = np.sum((pred_labels-1)*(true_labels-1),axis=2)

        return np.count_nonzero(result,axis=0).astype('float32')

    def FP(self,true_labels, pred_labels,mod='attr'):
        if mod == 'attr':
            result = pred_labels*(true_labels-1)
        else:
            result = np.sum(pred_labels*(true_labels-1),axis=2)
        return np.count_nonzero(result, axis=0).astype('float32')

    def FN(self,true_labels, pred_labels, mod ='attr'):
        if mod == "attr":
            result = (pred_labels-1)*true_labels
        else:
            result = np.sum((pred_labels-1)*true_labels, axis=2)

        return np.count_nonzero(result,axis=0).astype('float32')

    def precision(self, TP, FP, flag):
        assert flag == 'macro' or flag == 'micro', 'Please enter right flag...'
        if flag == 'macro':
            tmp = np.nonzero((np.sum(TP, axis=0) + np.sum(FP, axis=0) == 0))
            res = np.sum(TP, axis=0, dtype='float32') / (np.sum(TP, axis=0, dtype='float32') + np.sum(FP, axis=0, dtype='float32'))
            res[tmp] = 1
            return res
        else:
            return np.sum(TP) / (np.sum(TP) + np.sum(FP) + 1e-10)

    def per_precision(self,TP,FP):
        return np.divide(TP,np.add(TP,FP)+1e-10)


    def recall(self, TP, FN, flag):
        assert flag == 'macro' or flag == 'micro', 'Please enter right flag...'
        if flag == 'macro':
            tmp = np.nonzero((np.sum(TP, axis=0) + np.sum(FN, axis=0) == 0))
            res = np.sum(TP, axis=0, dtype='float32') / (
            np.sum(TP, axis=0, dtype='float32') + np.sum(FN, axis=0, dtype='float32'))
            res[tmp] = 1
            return res
        else:
            return np.sum(TP) / (np.sum(TP) + np.sum(FN) + 1e-10)

    def per_recall(self,TP, FN):
        return np.divide(TP, np.add(TP,FN)+1e-10)

    def f1_score(self, precision, recall, flag):
        assert flag == 'macro' or flag == 'micro', 'Please enter right flag...'
        if flag == 'macro':
            tmp = np.nonzero((precision + recall) == 0)
            res = 2 * precision * recall / (precision + recall + 1e-10)
            res[tmp] = 0
            return res
        else:
            return 2 * precision * recall / (precision + recall + 1e-10)

    def per_f1_score(self,per_precision, per_recall):
        return 2*per_precision*per_recall/(per_precision+per_recall+1e-10)

    def report(self,info,file=None,mod='std'):
        if mod=='std':
            print(info)
        else:
            file.write(info+'\n')