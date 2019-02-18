import numpy as np
import pandas as pd
import gensim
from nltk.corpus import stopwords
import string
import pickle
import os

class Dataset:
    def __init__(self,attr_labels,senti_labels, sentences,**kwargs):
        self.dataset_len=len(attr_labels)
        if len(kwargs)==0:
            self.batch_size=self.dataset_len
        else:
            self.batch_size=kwargs['batch_size']
        self.attr_labels=attr_labels
        self.senti_labels = senti_labels
        self.sentences=sentences
        self.count=0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count<self.dataset_len:
            if self.count+self.batch_size<self.dataset_len:
                attr_labels_batch= self.attr_labels[self.count:self.count+self.batch_size]
                senti_labels_batch = self.senti_labels[self.count:self.count+self.batch_size]
                sentences_batch = self.sentences[self.count:self.count+self.batch_size]
            else:
                attr_labels_batch = self.attr_labels[self.count:]
                senti_labels_batch = self.senti_labels[self.count:]
                sentences_batch=self.sentences[self.count:]
            self.count+=self.batch_size
        else:
            raise StopIteration
        return attr_labels_batch,senti_labels_batch,sentences_batch

class DataFeeder:
    def __init__(self, config):
        self.data_config = {
                            'train_data_file_path': '/hdd/lujunyu/dataset/meituan/train.pkl',
                            'test_data_file_path': '/hdd/lujunyu/dataset/meituan/dev.pkl',
                            'batch_size':1
                          }
        self.data_config.update(config)
        self.train_attr_labels, self.train_senti_labels, self.train_sentences,self.aspect_dic , self.dictionary, self.table = self.load_train_data()
        self.id_to_aspect_dic = dict((v,k) for k,v in self.aspect_dic.items())
        self.test_attr_labels, self.test_senti_labels, self.test_sentences = self.load_test_data()
        self.train_sentences, self.train_attr_labels, self.train_senti_labels = self.unison_shuffled_copies(self.train_sentences, self.train_attr_labels, self.train_senti_labels)
        print('#################################')
        print('train.shape: ',self.train_sentences.shape)
        print('test.shape: ',self.test_sentences.shape)
        print('#################################')
        self.train_data_size = len(self.train_attr_labels)
        self.test_data_size = len(self.test_attr_labels)

    def data_generator(self,flag):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch. 
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """
        # [( emb_id,fname,row_index m_id,c_id,typeText)]

        if flag == 'train':
            dataset = Dataset(self.train_attr_labels, self.train_senti_labels, self.train_sentences,batch_size=self.data_config['batch_size'])
        else:
            dataset = Dataset(self.test_attr_labels, self.test_senti_labels, self.test_sentences, batch_size=self.data_config['batch_size'])
        # during validation and test, to avoid errors are counted repeatedly,
        # we need to avoid the same data sended back repeately
        return dataset


    def unison_shuffled_copies(self, a, b, c):
        """
        shuffle a and b, and at the same time, keep their orders.
        :param a: 
        :param b: 
        :return: 
        """
        assert len(a) == len(b)
        assert len(a) == len(c)
        p = np.random.permutation(len(a))
        # p = np.arange(len(a))
        return a[p], b[p], c[p]

    def load_train_data(self):
        if os.path.exists(self.data_config['train_data_file_path']) and os.path.getsize(self.data_config['train_data_file_path']) > 0:
            with open(self.data_config['train_data_file_path'],'rb') as f:
                attribute_dic, word_dic, attr_labels, senti_labels, sentence, word_embed = pickle.load(f)

            return attr_labels, senti_labels, sentence , attribute_dic , word_dic ,word_embed

    def load_test_data(self):
        print('test path: ',self.data_config['test_data_file_path'])
        if os.path.exists(self.data_config['test_data_file_path']) and os.path.getsize(self.data_config['test_data_file_path']) > 0:
            with open(self.data_config['test_data_file_path'],'rb') as f:
                attr_labels, senti_labels, sentence = pickle.load(f)
            return attr_labels, senti_labels, sentence




