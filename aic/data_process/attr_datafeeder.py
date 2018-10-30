import numpy as np
import pandas as pd
import gensim
from nltk.corpus import stopwords
import string
import pickle
import os

class Dataset:
    def __init__(self,labels,sentences,**kwargs):
        self.dataset_len=len(labels)
        if len(kwargs)==0:
            self.batch_size=self.dataset_len
        else:
            self.batch_size=kwargs['batch_size']
            self.dict = kwargs['dict']
        self.labels=labels
        self.sentences=sentences
        self.count=0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count<self.dataset_len:
            if self.count+self.batch_size<self.dataset_len:
                labels_batch=self.labels[self.count:self.count+self.batch_size]
                sentences_batch = self.sentences[self.count:self.count+self.batch_size]
            else:
                labels_batch=self.labels[self.count:]
                sentences_batch=self.sentences[self.count:]
            self.count+=self.batch_size
            sentences_batch = self.sent2batch(sentences_batch, self.dict)
        else:
            raise StopIteration
        return labels_batch,sentences_batch

    def sent2batch(self, batch, word_dic):
        res = []
        for text in batch:
            tmp = []
            for sent in text:
                tmp.append([word_dic[x] for x in sent.split()])
            res.append(np.array(tmp))
        return np.array(res)

class DataFeeder():
    def __init__(self, config):
        self.data_config = {
                            'train_data_path': '/hdd/lujunyu/dataset/meituan/train.pkl',
                            'test_data_path': '/hdd/lujunyu/dataset/meituan/dev.pkl',
                            'top_k_data':10,
                            'batch_size':2,
                          }
        self.data_config.update(config)
        self.train_labels, self.train_sentences,self.aspect_dic , self.dictionary,self.table = self.load_train_data()
        self.test_labels, self.test_sentences = self.load_test_data()
        self.train_sentences, self.train_labels = self.unison_shuffled_copies(self.train_sentences, self.train_labels)
        print('train.shape: ',self.train_sentences.shape)
        print('test.shape: ',self.test_sentences.shape)
        self.train_data_size = len(self.train_labels)
        self.test_data_size = len(self.test_labels)

    def data_generator(self,flag):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch. 
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """
        # [( emb_id,fname,row_index m_id,c_id,typeText)]

        if flag == 'train':
            dataset = Dataset(self.train_labels,self.train_sentences,batch_size=self.data_config['batch_size'],dict=self.dictionary)
        else:
            dataset = Dataset(self.test_labels,self.test_sentences, batch_size=self.data_config['batch_size'], dict=self.dictionary)
        # during validation and test, to avoid errors are counted repeatedly,
        # we need to avoid the same data sended back repeately
        return dataset


    def unison_shuffled_copies(self, a, b):
        """
        shuffle a and b, and at the same time, keep their orders.
        :param a: 
        :param b: 
        :return: 
        """
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def load_train_data(self):
        with open(self.data_config['train_data_path'],'rb') as f:
            attribute_dic, word_dic,label,_, sentence, word_embed  = pickle.load(f)

        return label, sentence , attribute_dic , word_dic ,word_embed

    def load_test_data(self):
        print('test path: ',self.data_config['test_data_path'])
        with open(self.data_config['test_data_path'],'rb') as f:
            label, _, sentence = pickle.load(f)
        return label, sentence

if __name__ == "__main__":
    data_config = {}
    df = DataFeeder(data_config)



