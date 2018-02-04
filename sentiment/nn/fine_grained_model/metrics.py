import numpy as np
import pickle
import numpy as np

class Metrics:
    def __init__(self,preds,labels,type_code,train_queue):
        index=[]
        self.train_queue = train_queue
        for t in train_queue:
            index.append(type_code[t])
        self.pred=[]
        self.label=[]
        for i in index:
            self.pred.append(preds[:,i])
            self.label.append(labels[:,i])

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

    def accuracy(self,tp,tn,fp,fn):
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

    def eval(self):
        TP=[]
        TN=[]
        FP=[]
        FN=[]
        P=[]
        preserve_P = []
        R=[]
        preserve_R = []
        A=[]
        F1 = []
        preserve_F1 = []
        for i in range(len(self.pred)):
            pred_i = self.pred[i]
            label_i = self.label[i]
            tp = self.tpos(pred_i,label_i)
            tn = self.tneg(pred_i,label_i)
            fp = self.fpos(pred_i,label_i)
            fn = self.fneg(pred_i, label_i)
            a=self.accuracy(tp,tn,fp,fn)
            p=self.precision(tp,tn,fp,fn)
            r=self.recall(tp,tn,fp,fn)
            f1=self.f1_score(r,p)
            TP.append(tp)
            TN.append(tn)
            FP.append(fp)
            FN.append(fn)
            if p!=-1:
                P.append(p)
            preserve_P.append(p)
            if r !=-1:
                R.append(r)
            preserve_R.append(r)
            A.append(a)
            if f1!=-1:
                F1.append(f1)
            preserve_F1.append(f1)
        if len(P)!=0:
            micro_p=self.micro_precision(TP,TN,FP,FN)
            macro_p = self.macro_precision(P)
        else:
            micro_p = -1
            macro_p = -1
        if len(R) !=0:
            micro_r=self.micro_recall(TP,TN,FP,FN)
            macro_r = self.macro_recall(R)
        else:
            micro_r = -1
            macro_r = -1
        if micro_r!=-1 and micro_p!=-1:
            micro_f1=self.micro_f1_score(micro_r=micro_r,micro_p=micro_p)
        else:
            micro_f1=-1
        if macro_r!=-1 and macro_p!=-1:
            macro_f1=self.macro_f1_score(macro_r=macro_r,macro_p=macro_p)
        else:
            macro_f1=-1

        dic={}
        for i in range(len(self.train_queue)):
            type_txt = self.train_queue[i]
            dic[type_txt]={'precision':preserve_P[i],'recall':preserve_R[i],'accuracy':A[i],'f1':preserve_F1[i]}

        return {'f1':{'macro':macro_f1,'micro':micro_f1},
                'p':{'macro':macro_p,'micro':micro_p},
                'r':{'macro':macro_r,'micro':micro_r}},dic

