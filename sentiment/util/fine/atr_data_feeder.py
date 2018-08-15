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

class DataFeeder():
    def __init__(self, config):
        self.config = config
        self.train_labels, self.train_sentences,self.aspect_dic , self.dictionary,self.table = self.load_train_data()
        self.test_labels, self.test_sentences = self.load_test_data()
        self.train_sentences, self.train_labels = self.unison_shuffled_copies(self.train_sentences, self.train_labels)
        self.train_labels,self.train_sentences = self.train_labels[:self.config['top_k_data']] , self.train_sentences[:self.config['top_k_data']]
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
            dataset = Dataset(self.train_labels,self.train_sentences,batch_size=self.config['batch_size'])
        else:
            dataset = Dataset(self.test_labels,self.test_sentences, batch_size=self.config['batch_size'])
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

    def table_generator(self,word_embed,word_list):
        """
        Generate word embedding matirx
        :return: word embeddings table: shape = (number of words, word dimension) word1 [0.223, -4.222, ....] word2 [0.883, 0.333, ....] ... ...
        """
        tmp = word_embed.syn0
        table = tmp[word_list]
        unk_vec = np.mean(tmp,axis=0).reshape(1,self.config['word_dim'])
        pad_vec = np.zeros((1,self.config['word_dim']))
        vec = np.append(unk_vec,pad_vec,axis=0)
        table = np.append(table,vec,axis=0)
        print("Generate table finished...")

        return table


    def get_aspect_id(self,data,aspect_dic):
        """
        Generate attribute ground truth
        :param data: 
        :param start: 
        :param end: 
        :return: shape = (batch size, attribute numbers) eg. [[1,0,1,...],[0,0,1,...],...]
        """
        aspect = []
        for i in np.arange(0,data.shape[0]):
            vec = np.zeros(len(aspect_dic))
            for j in data[data['sentence_id'] == data.iloc[i]['sentence_id']]['category'].unique():
                if j == j:
                    vec[aspect_dic[j]] = 1
            aspect.append(vec)
        aspect = np.array(aspect)
        if np.nan in aspect_dic.keys():
            aspect  = np.delete(aspect,obj=aspect_dic[np.nan],axis=1)
        return aspect

    def get_sentiment_id(self,data,aspect_dic,sent_dic):
        """
        Generate sentiment ground truth
        :param data: 
        :param start: 
        :param end: 
        :return:  shape =(batch size, attributes number +1, 3) 
        """

        sentiment = []
        for i in np.arange(0,data.shape[0]):
            vec = []
            for j in aspect_dic.keys():
                if j in data[data['sentence_id'] == data.iloc[i]['sentence_id']]['category'].unique() and sent_dic[data.iloc[i]['polarity']]<3:
                    tmp = np.zeros(3)
                    tmp[sent_dic[data.iloc[i]['polarity']]] = 1
                    vec.append(tmp)
                else:
                    vec.append(np.zeros(3))
            sentiment.append(np.array(vec))
        return np.array(sentiment)

    def get_word_id(self,data,dictionary):
        """
        Generate sentences id matrix
        :param data:
        :param start:
        :param end:
        :return: shape = (batch size, words number) eg. [[1,4,6,8,...],[7,1,5,10,...],]
        """

        punctuation = '!"#&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        outtab = ' ' * len(punctuation)
        intab = punctuation
        trantab = str.maketrans(intab, outtab)
        sentence = []
        for i in range(data.shape[0]):
            a = []
            num = 0
            for word in data.iloc[i]['text'].lower().translate(trantab).split():
                if num >= self.config['words_num']:
                    break
                a.append(dictionary[word])
                num += 1
            if num < self.config['words_num']:
                for j in range(self.config['words_num'] - num):
                    a.append(self.config['padding_word_index'])
            sentence.append(np.array(a))
        return np.array(sentence)

    def load_train_data(self):
        if os.path.exists(self.config['train_data_file_path']) and os.path.getsize(self.config['train_data_file_path']) > 0:
            with open(self.config['train_data_file_path'],'rb') as f:
                attribute_dic = pickle.load(f)
                word_dic = pickle.load(f)
                label = pickle.load(f)
                sentence = pickle.load(f)
                word_embed = pickle.load(f)

        return label, sentence , attribute_dic , word_dic ,word_embed

    def load_test_data(self):
        if os.path.exists(self.config['test_data_file_path']) and os.path.getsize(self.config['test_data_file_path']) > 0:
            with open(self.config['test_data_file_path'],'rb') as f:
                label = pickle.load(f)
                sentence = pickle.load(f)

        return label, sentence




