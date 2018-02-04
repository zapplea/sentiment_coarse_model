import numpy as np
import pickle
import numpy as np

class Metrics:
    def __init__(self,preds,labels,nn_config):
        self.preds = preds
        self.labels =labels
        self.nn_config = nn_config

    def micro_f1_score(self,micro_r,micro_p):
        return 2*micro_p*micro_r/(micro_p+micro_r)

    def macro_f1_score(self,macro_r,macro_p):
        return 2*macro_r*macro_p/(macro_r+macro_p)

    def micro_precision(self,TP,TN,FP,FN):
        return np.sum(TP)/(np.sum(TP)+np.sum(FP))

    def macro_precision(self,P):
        return np.mean(P)

    def micro_recall(self,TP,TN,FP,FN):
        return np.sum(TP)/(np.sum(TP)+np.sum(FN))

    def macro_recall(self,R):
        return np.mean(R)

    def tpos(self,pred,label):
        count=0
        for i in range(label.shape[0]):
            if label[i] == 1 and pred[i] ==1:
               count+=1
        return count

    def tneg(self,pred,label):
        count = 0
        for i in range(label.shape[0]):
            if label[i] == 0 and pred[i] ==0:
               count+=1
        return count

    def fpos(self,pred,label):
        count = 0
        for i in range(label.shape[0]):
            if label[i] == 0 and pred[i] == 1:
                count += 1
        return count

    def fneg(self,pred,label):
        count = 0
        for i in range(label.shape[0]):
            if label[i] == 1 and pred[i] == 0:
                count += 1
        return count

    def specific_atr_accuracy(self,tp,tn,fp,fn):
        return (tp+tn)/(tp+tn+fp+fn)

    def precision(self,tp,tn,fp,fn):
        if tp ==0 and fp==0:
            return -1
        else:
            return tp/(tp+fp)

    def recall(self,tp,tn,fp,fn):
        if tp ==0 and fn ==0:
            return -1
        else:
            return tp/(tp+fn)

    def f1_score(selfs,r,p):
        if p==-1 or r == -1:
            return -1
        elif p==0 and r==0:
            return -1
        else:
            return 2*p*r/(p+r)

    def accuracy(self,preds,labels):
        #
        condition = np.greater(labels, self.nn_config['label_atr_threshold'])
        labels = np.where(condition,np.ones_like(labels,dtype='float32'),np.zeros_like(labels,dtype='float32'))
        #
        condition = np.equal(preds,labels)
        temp = np.where(condition,np.zeros_like(preds,dtype='float32'),np.ones_like(preds,dtype='float32'))
        temp=np.sum(temp,axis=1)
        condition = np.equal(temp,np.zeros_like(temp,dtype='float32'))
        temp = np.where(condition, np.ones_like(temp,dtype='float32'),np.zeros_like(temp,dtype='float32'))
        return np.mean(temp)

    def eval(self):
        accuracy = self.accuracy(self.preds,self.labels)
        return accuracy

if __name__ == "__main__":
    preds = np.array([])
    labels = np.array()
    Metrics()