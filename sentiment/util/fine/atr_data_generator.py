import numpy as np
import pandas as pd
import gensim
from nltk.corpus import stopwords
import string
import pickle
import os

class DataGenerator():
    def __init__(self, configs):
        self.configs = configs
        self.train_label, self.train_sentence,self.aspect_dic , self.dictionary,self.table = self.load_train_data()
        self.test_label, self.test_sentence = self.load_test_data(self.aspect_dic,self.dictionary)
        self.train_sentence, self.train_label = self.unison_shuffled_copies(self.train_sentence, self.train_label)
        self.train_label,self.train_sentence = self.train_label[:self.configs['top_k_data']] , self.train_sentence[:self.configs['top_k_data']]
        self.train_data_size = len(self.train_label)
        self.test_data_size = len(self.test_label)


    def train_data_generator(self,batch_num):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch. 
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """
        # [( emb_id,fname,row_index m_id,c_id,typeText)]
        #
        # if batch_num == 0:
        #     self.train_sentence, self.train_label = self.unison_shuffled_copies(self.train_sentence,self.train_label)
        train_size = self.train_data_size
        start = batch_num * self.configs['batch_size'] % train_size
        end = (batch_num * self.configs['batch_size'] + self.configs['batch_size']) % train_size
        if start < end:
            batches_label = self.train_label[start:end]
            batches_sentence = self.train_sentence[start:end]
        else:
            batches_label = self.train_label[
                                             train_size - self.configs['batch_size']:train_size]
            batches_sentence = self.train_sentence[
                                            train_size - self.configs['batch_size']:train_size]
        # during validation and test, to avoid errors are counted repeatedly,
        # we need to avoid the same data sended back repeately
        return batches_sentence, batches_label

    def test_data_generator(self):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch. 
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """
        return self.test_sentence,self.test_label

    def unison_shuffled_copies(self, a, b):
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
        unk_vec = np.mean(tmp,axis=0).reshape(1,self.configs['word_dim'])
        pad_vec = np.zeros((1,self.configs['word_dim']))
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
                if num >= self.configs['words_num']:
                    break
                a.append(dictionary[word])
                num += 1
            if num < self.configs['words_num']:
                for j in range(self.configs['words_num'] - num):
                    a.append(self.configs['padding_word_index'])
            sentence.append(np.array(a))
        return np.array(sentence)

    # def get_word_id(self,data,dictionary):
    #     """
    #     Generate sentences id matrix
    #     :param data:
    #     :param start:
    #     :param end:
    #     :return: shape = (batch size, words number) eg. [[1,4,6,8,...],[7,1,5,10,...],]
    #     """
    #
    #     sentence = []
    #     for sent in data['normalization']:
    #         a = []
    #         num = 0
    #         for word in sent:
    #             if num >= self.configs['words_num']:
    #                 break
    #             a.append(dictionary[word])
    #             num += 1
    #         if num < self.configs['words_num']:
    #             for j in range(self.configs['words_num'] - num):
    #                 a.append(dictionary['#PAD#'])
    #         sentence.append(np.array(a))
    #     return np.array(sentence)


    def load_train_data(self):
        if os.path.exists(self.configs['fine_train_data_file']) and os.path.getsize(self.configs['fine_train_data_file']) > 0:
            with open(self.configs['fine_train_data_file'],'rb') as f:
                attribute_dic, word_dic, label, sentence, word_embed = pickle.load(f)
        else:
            with open(self.configs['fine_train_source_file'],'rb') as f:
                tmp = pickle.load(f)

            ##Generate attribute_dic
            attribute_dic = {}
            i = 0
            for attribute in tmp['category'].unique():
                if attribute == attribute:
                    attribute_dic[attribute] = i
                    i += 1


            ###Generate dictionary
            with open(self.configs['dictionary'],'rb') as wd:
                word_list, word_dic, word_embed = pickle.load(wd)
                print('The number of words in dataset:',len(word_dic))

            label = self.get_aspect_id(tmp,attribute_dic)
            train_data_mask = tmp['sentence_id'].drop_duplicates().index
            label = label[train_data_mask]
            sentence = self.get_word_id(tmp,word_dic)
            sentence = sentence[train_data_mask]


            with open(self.configs['fine_train_data_file'],'wb') as f:
                pickle.dump((attribute_dic, word_dic, label, sentence, word_embed), f)

        return label, sentence , attribute_dic , word_dic ,word_embed

    def load_test_data(self,test_attribute_dic,dictionary):
        if os.path.exists(self.configs['fine_test_data_file']) and os.path.getsize(self.configs['fine_test_data_file']) > 0:
            with open(self.configs['fine_test_data_file'],'rb') as f:
                label, sentence = pickle.load(f)
        else:
            with open(self.configs['fine_test_source_file'], 'rb') as f:
                tmp = pickle.load(f)
            test_data_mask = tmp['sentence_id'].drop_duplicates().index
            label = self.get_aspect_id(tmp,test_attribute_dic)
            label = label[test_data_mask]
            sentence = self.get_word_id(tmp,dictionary)
            sentence = sentence[test_data_mask]

            with open(self.configs['fine_test_data_file'], 'wb') as f:
                pickle.dump((label,sentence), f)
        return label, sentence




