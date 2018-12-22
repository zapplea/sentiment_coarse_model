import numpy as np

class Metrics:
    def __init__(self,config):
        self.config = config

    def caliberate(self,label):
        """
        [pos, neu, neg] --> [pos, neu, neg, not mention]
        :param label: (batch size, attributes+1,3) 
        :return: 
        """
        notmention_label = np.expand_dims(np.sum(label,axis=2),axis=2)
        condition = np.equal(notmention_label,np.zeros_like(notmention_label))
        notmention_label = np.where(condition,np.ones_like(notmention_label),np.zeros_like(notmention_label))
        # shape = (batch size, attributes+1,4)
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



    def calculate_metrics_score(self,TP_vec, FP_vec, FN_vec,outf,id_to_aspect_dic, mod='attr'):
        _precision = self.macro_precision(TP_vec, FP_vec)
        _recall = self.macro_recall(TP_vec, FN_vec)

        per_precision = self.per_precision(TP_vec, FP_vec)
        per_recall = self.per_recall(TP_vec, FN_vec)
        _f1_score = self.macro_f1_score(per_precision, per_recall)
        self.report('Macro F1 score: %.5f\nMacro precision:%.5f\nMacro recall:%.5f'
                       % (_f1_score, _precision, _recall), outf, 'report')

        # per class metrics
        per_f1_score = self.per_f1_score(per_precision, per_recall)
        if mod == 'attr':
            for i in range(per_f1_score.shape[0]):
                self.report('%s f1 score: %.5f          precision: %.5f           recall: %.5f'
                               % (id_to_aspect_dic[i], per_f1_score[i], per_precision[i], per_recall[i]), outf,
                               'report')
        else:
            per_precision = np.reshape(per_precision,
                                      newshape=(self.config['attributes_num'], 4))
            per_recall = np.reshape(per_recall,
                                    newshape=(self.config['attributes_num'], 4))
            per_f1_score = np.reshape(per_f1_score,newshape=(self.config['attributes_num'], 4))
            sentiment=['pos','neu','neg','nmt']
            for i in range(per_f1_score.shape[0]):
                for j in range(per_f1_score.shape[1]):
                    self.report('%s.%s f1 score: %.5f          precision: %.5f         recall: %.5f'
                                % (id_to_aspect_dic[i],sentiment[j], per_f1_score[i][j], per_precision[i][j], per_recall[i][j]), outf,
                                'report')
        return _f1_score