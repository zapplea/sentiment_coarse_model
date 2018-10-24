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
        else:
            raise StopIteration
        return labels_batch,sentences_batch

# TODO: check whether coarse and fine use the same wordembedding.
class DataFeeder():
    def __init__(self, config):
        self.data_config = {
                            'top_k_data':2000,
                            'train_data_file_path': '/datastore/liu121/sentidata2/expdata/aic2018/fine_data/train_fine.pkl',
                            'test_data_file_path': '/datastore/liu121/sentidata2/expdata/aic2018/fine_data/dev_fine.pkl',
                            'wordembedding_file_path': '/datastore/liu121/wordEmb/googlenews/GoogleNews-vectors-negative300.bin',
                          }
        self.data_config.update(config)
        self.train_labels, self.train_sentences,self.aspect_dic , self.dictionary,self.table = self.load_train_data()
        self.test_labels, self.test_sentences = self.load_test_data()
        self.train_sentences, self.train_labels = self.unison_shuffled_copies(self.train_sentences, self.train_labels)
        self.train_labels,self.train_sentences = self.train_labels[:self.data_config['top_k_data']] , self.train_sentences[:self.data_config['top_k_data']]
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
            dataset = Dataset(self.train_labels,self.train_sentences,batch_size=self.data_config['batch_size'])
        else:
            dataset = Dataset(self.test_labels,self.test_sentences, batch_size=self.data_config['batch_size'])
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
        if os.path.exists(self.data_config['train_data_file_path']) and os.path.getsize(self.data_config['train_data_file_path']) > 0:
            with open(self.data_config['train_data_file_path'],'rb') as f:
                data=pickle.load(f)
                attribute_dic = data[0]
                word_dic = data[1]
                label = data[2]
                print(label.shape)
                sent = data[3]
                print(sent.shape)
                sentence = data[4]
                print(sentence.shape)
                print('========')
                word_embed = data[5]

            return label, sentence, attribute_dic, word_dic, word_embed

    def load_test_data(self):
        print('test path: ',self.data_config['test_data_file_path'])
        if os.path.exists(self.data_config['test_data_file_path']) and os.path.getsize(self.data_config['test_data_file_path']) > 0:
            with open(self.data_config['test_data_file_path'],'rb') as f:
                data = pickle.load(f)
                label = data[0]
                print(label.shape)
                sent = data[1]
                print(sent.shape)
                sentence = np.array(data[2],dtype='float32')
                print('sentence type: ',type(sentence))
                print('sentence1: ',sentence[0])
            return label, sentence

if __name__ == "__main__":
    data_config = {}
    df = DataFeeder(data_config)



