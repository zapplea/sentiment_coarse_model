import tensorflow as tf
import numpy as np

class Metrics:
    def caliberate(self,label):
        notmention_label = tf.tile(np.sum(label,axis=2),axis=2)
        condition = np.equal(notmention_label,np.zeros_like(notmention_label))
        notmention_label = np.where(condition,np.ones_like(notmention_label),np.zeros_like(notmention_label))
        label = np.concatenate([label,notmention_label],axis=2)
        d0=label.shape[0]
        return np.reshape(label,newshape=[d0,-1])

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

    def macro_precision(self, TP, FP):
        return np.mean(TP / (TP + FP + 1e-10))

    def per_precision(self,TP,FP):
        return np.divide(TP,np.add(TP,FP)+1e-10)


    def macro_recall(self, TP, FN):
        return np.mean(TP / (TP + FN + 1e-10))

    def per_recall(self,TP, FN):
        return np.divide(TP, np.add(TP,FN)+1e-10)

    def macro_f1_score(self, per_precision, per_recall):
        return np.mean(2*per_precision*per_recall/(per_precision+per_recall+1e-10))


    def per_f1_score(self,per_precision, per_recall):
        return np.divide(2*per_precision*per_recall,(per_precision+per_recall+1e-10))

    def report(self,info,file=None,mod='std'):
        if mod=='std':
            print(info)
        else:
            file.write(info+'\n')
            file.flush()



    def calculate_metrics_score(self,TP_vec, FP_vec, FN_vec,outf,id_to_aspect_dic):
        _precision = self.macro_precision(TP_vec, FP_vec)
        _recall = self.macro_recall(TP_vec, FN_vec)

        per_precision = self.per_precision(TP_vec, FP_vec)
        per_recall = self.per_recall(TP_vec, FN_vec)
        _f1_score = self.macro_f1_score(per_precision, per_recall)
        self.report('Macro F1 score: %.10f\nMacro precision:%.10f\nMacro recall:%.10f'
                       % (_f1_score, _precision, _recall), outf, 'report')

        # per class metrics
        per_f1_score = self.per_f1_score(per_precision, per_recall)
        for i in range(per_precision.shape[0]):
            self.report('%s f1 score: %.10f precision: %.10f recall: %.10f'
                           % (id_to_aspect_dic[i], per_f1_score[i], per_precision[i], per_recall[i]), outf,
                           'report')
        return _f1_score