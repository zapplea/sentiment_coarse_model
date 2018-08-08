import numpy as np
import pandas as pd
import gensim
from nltk.corpus import stopwords
import string
import os
import pickle
import nltk

class DataGenerator():
    def __init__(self,configs):
        self.configs = configs
        self.train_label, self.train_sentence,self.aspect_dic , self.dictionary,self.table = self.load_train_data()
        print(self.aspect_dic)
        exit()
        self.test_label, self.test_sentence = self.load_test_data(self.aspect_dic,self.dictionary)
        self.train_data_size = len(self.train_label)
        self.val_data_size = len(self.test_label)


    def data_generator(self,batch_num,flag):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch. 
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """
        # [( emb_id,fname,row_index m_id,c_id,typeText)]

        if flag == 'train':
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
        else:
            val_size = self.val_data_size
            start = batch_num * self.configs['batch_size'] % val_size
            end = (batch_num * self.configs['batch_size'] + self.configs['batch_size']) % val_size
            if start < end:
                batches_label = self.train_label[start:end]
                batches_sentence = self.train_sentence[start:end]
            else:
                batches_label = self.train_label[
                                                 val_size - self.configs['batch_size']:val_size]
                batches_sentence = self.train_sentence[
                                                val_size - self.configs['batch_size']:val_size]
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


    def get_aspect_probility(self,data):
        """
        Generate attribute ground truth
        :param data: 
        :param start: 
        :param end: 
        :return: shape = (batch size, attribute numbers) eg. [[1,0,1,...],[0,0,1,...],...]
        """
        aspect = data['Y_att'].tolist()
        return np.array(aspect)

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

        review = []
        punctuation = '!"#&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        outtab = ' ' * len(punctuation)
        intab = punctuation
        trantab = str.maketrans(intab, outtab)
        for idx in range(data.shape[0]):
            sens = nltk.sent_tokenize(data.iloc[idx]['text'])
            if len(sens) > self.configs['max_review_length']:
                sens = sens[:self.configs['max_review_length']]
            sentence = []
            for sent in sens:
                a = []
                num = 0
                for word in sent.lower().translate(trantab).split():
                    if num >= self.configs['words_num']:
                        break
                    a.append(dictionary[word])
                    num += 1
                if num < self.configs['words_num']:
                    for j in range(self.configs['words_num'] - num):
                        a.append(self.configs['padding_word_index'])
                else:
                    a = a[:self.configs['words_num']]

                sentence.append(np.array(a))
            if len(sentence) < self.configs['max_review_length']:
                for j in range(self.configs['max_review_length'] - len(sentence)):
                    sentence.append(np.array([dictionary['#PAD#']] * self.configs['words_num']))
            else:
                sentence = sentence[:self.configs['max_review_length']]
            review.append(np.array(sentence))
        return np.array(review)

    # def get_word_id(self, data, w2id):
    #     """
    #     Generate sentences id matrix
    #     :param data:
    #     :param start:
    #     :param end:
    #     :return: shape = (batch size, words number) eg. [[1,4,6,8,...],[7,1,5,10,...],]
    #     """
    #
    #     review = []
    #     for sens in data['review_normalization']:
    #         if len(sens) > self.configs['max_review_length']:
    #             sens = sens[:self.configs['max_review_length']]
    #         sentence = []
    #         for sent in sens:
    #             tmp = []
    #             for word in sent:
    #                 tmp.append(w2id[word])
    #             if len(tmp) < self.configs['words_num']:
    #                 for j in range(self.configs['words_num'] - len(tmp)):
    #                     tmp.append(w2id['#PAD#'])
    #             else:
    #                 tmp = tmp[:self.configs['words_num']]
    #
    #             sentence.append(np.array(tmp))
    #         if len(sentence) < self.configs['max_review_length']:
    #             for j in range(self.configs['max_review_length'] - len(sentence)):
    #                 sentence.append(np.array([w2id['#PAD#']] * self.configs['words_num']))
    #         else:
    #             sentence = sentence[:self.configs['max_review_length']]
    #         review.append(np.array(sentence))
    #     return np.array(review)

    def get_word_list(self, data ,word_dic):
        outtab = ' ' * len(string.punctuation)
        intab = string.punctuation
        trantab = str.maketrans(intab, outtab)
        word_list = []
        for i in np.arange(0, data.shape[0]):
            for word in data.iloc[i]['text'].lower().translate(trantab).split():
                if word in word_dic.keys():
                    word_list.append(word_dic[word])
        word_list = list(set(word_list))
        print('The number of words in dataset:',len(word_list))
        return word_list


    def load_train_data(self):
        print('coarse_test_data_file: ', self.configs['coarse_test_data_file'])
        print('coarse_test_data_file: ', self.configs['coarse_test_data_file'])
        if os.path.exists(self.configs['coarse_train_data_file']) and os.path.getsize(self.configs['coarse_train_data_file']) > 0:
            with open(self.configs['coarse_train_data_file'],'rb') as f:
                aspect_dic, word_dic, label, sentence, word_embed = pickle.load(f)
        else:
            with open(self.configs['coarse_train_source_file'],'rb') as f:
                tmp = pickle.load(f)

            ##Generate attribute_dic
            aspect_dic ={'RESTAURANT': 0, 'SERVICE': 1, 'FOOD': 2
            , 'DRINKS': 3, 'AMBIENCE': 4, 'LOCATION': 5,'OTHER':6}

            ###Generate word_dic
            with open(self.configs['dictionary'],'rb') as wd:
                word_list, word_dic, word_embed = pickle.load(wd)
                print('The number of words in dataset:',len(word_dic))

            label = self.get_aspect_probility(tmp)
            train_data_mask = tmp['text'].drop_duplicates().reset_index().index
            label = label[train_data_mask]
            sentence = self.get_word_id(tmp,word_dic)
            sentence = sentence[train_data_mask]


            ###Generate word_embed

            with open(self.configs['coarse_train_data_file'], 'wb') as f:
                pickle.dump((aspect_dic, word_dic, label, sentence, word_embed), f)

        return label, sentence , aspect_dic , word_dic ,word_embed

    def load_test_data(self,test_aspect_dic,word_dic):
        if os.path.exists(self.configs['coarse_test_data_file']) and os.path.getsize(self.configs['coarse_test_data_file']) > 0:
            f = open(self.configs['coarse_test_data_file'],'rb')
            label = pickle.load(f)
            sentence  = pickle.load(f)
            f.close()
        else:
            f = open(self.configs['coarse_test_data_file'], 'wb')
            with open(self.configs['coarse_test_source_file'],'rb') as ff:
                tmp = pickle.load(ff)
            test_data_mask = tmp['text'].drop_duplicates().reset_index().index
            label = self.get_aspect_probility(tmp)
            label = label[test_data_mask]
            pickle.dump(label, f)
            sentence = self.get_word_id(tmp,word_dic)
            sentence = sentence[test_data_mask]
            pickle.dump(sentence, f)
            f.close()
        return label, sentence

    def fine_sentences(self,attribute,sentence):
        if os.path.exists(self.configs['fine_sentences_file']):
            with open(self.configs['fine_sentences_file'],'rb') as f:
                fine_sent = pickle.load(f)
        else:
            # with open(self.configs['fine_sentences_file'],'wb') as f:
            # r = np.repeat(np.reshape(attribute.transpose()[0],[1995,1]),axis=1,repeats=40) * sentence
            # print(r[~np.all(r == 0, axis=1)].astype(int).shape)
            fine_sent = []
            for atr_vec in attribute.transpose():
                r = np.repeat(np.reshape(atr_vec, [2000, 1]), axis=1, repeats=40) * sentence
                r = np.reshape(r[~np.all(r == 0, axis=1)].astype(int),[-1,1,40])
                fine_sent.append(r)
            fine_sent = np.array(fine_sent)
            with open(self.configs['fine_sentences_file'], 'wb') as f:
                pickle.dump(fine_sent,f)
        return fine_sent




