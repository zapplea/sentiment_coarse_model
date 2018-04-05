import numpy as np
import pandas as pd
import gensim
from nltk.corpus import stopwords
import string
import os
import pickle

class DataGenerator():
    def __init__(self,data_config):
        self.data_config = data_config
        # self.word_embed = gensim.models.KeyedVectors.load_word2vec_format(self.data_config['wordembedding_file_path'], binary=True, unicode_errors='ignore')
        # self.vocabulary = self.word_embed.index2word
        # self.dictionary = {}
        # for i in range(len(self.vocabulary)):
        #     self.dictionary[self.vocabulary[i]] = i


        self.aspect_freq= self.load_train_data()
        # self.load_test_data(self.aspect_dic,)

    def get_aspect_freq(self,data,aspect_dic):
        """
        Generate attribute ground truth
        :param data: 
        :param start: 
        :param end: 
        :return: shape = (batch size, attribute numbers) eg. [[1,0,1,...],[0,0,1,...],...]
        """
        aspect_freq={}
        for key in aspect_dic:
            aspect_freq[key] = {'key':aspect_dic[key],'freq':0,'ratio':0}
        aspect = []
        for i in np.arange(0,data.shape[0]):
            vec = np.zeros(len(aspect_dic))
            for j in data[data['sentence_id'] == data.iloc[i]['sentence_id']]['category'].unique():
                    vec[aspect_dic[j]] = 1
                    aspect_freq[j]['freq']+=1
            aspect.append(vec)
        sum=0
        for key in aspect_freq:
            sum+=aspect_freq[key]['freq']
        for key in aspect_freq:
            aspect_freq[key]['ratio'] = aspect_freq[key]['freq']/sum
        return aspect_freq

    # def get_sentiment_id(self,data,aspect_dic,sent_dic):
    #     """
    #     Generate sentiment ground truth
    #     :param data:
    #     :param start:
    #     :param end:
    #     :return:  shape =(batch size, attributes number +1, 3)
    #     """
    #     sentiment = []
    #     for i in np.arange(0,data.shape[0]):
    #         vec = []
    #         for j in aspect_dic.keys():
    #             if j in data[data['sentence_id'] == data.iloc[i]['sentence_id']]['category'].unique() and sent_dic[data.iloc[i]['polarity']]<3:
    #                 tmp = np.zeros(3)
    #                 tmp[sent_dic[data.iloc[i]['polarity']]] = 1
    #                 vec.append(tmp)
    #             else:
    #                 vec.append(np.zeros(3))
    #         sentiment.append(np.array(vec))
    #     return np.array(sentiment)

    # def get_word_id(self,data):
    #     """
    #     Generate sentences id matrix
    #     :param data:
    #     :param start:
    #     :param end:
    #     :return: shape = (batch size, words number) eg. [[1,4,6,8,...],[7,1,5,10,...],]
    #     """
    #
    #     outtab = ' ' * len(string.punctuation)
    #     intab = string.punctuation
    #     trantab = str.maketrans(intab, outtab)
    #     stopWords = set(pd.read_table(self.data_config['stopwords_file_path'],header=None,delimiter="\t")[0]) | set(stopwords.words('english'))
    #
    #     sentence = []
    #     for i in np.arange(0,data.shape[0]):
    #         a = []
    #         num = 0
    #         for word in data.iloc[i]['sentence'].lower().translate(trantab).split():
    #             if num >= self.data_config['words_num']:
    #                 break
    #             if word not in stopWords and word in self.word_embed.index2word:
    #                 a.append(self.dictionary[word])
    #                 num += 1
    #         if num < self.data_config['words_num']:
    #             for j in range(self.data_config['words_num'] - num):
    #                 a.append(self.data_config['padding_word_index'])
    #         sentence.append(np.array(a))
    #     return np.array(sentence)

    def load_train_data(self):
        if os.path.exists(self.data_config['train_data_file_path']) and os.path.getsize(self.data_config['train_data_file_path']) > 0:
            f = open(self.data_config['train_data_file_path'],'rb')
            attribute_ground_truth = pickle.load(f)
            sentence_ground_truth  = pickle.load(f)
            train_aspect_dic = pickle.load(f)
            f.close()
        else:
            tmp = pd.read_csv(self.data_config['train_source_file_path'], index_col=0)
            train_aspect_dic = {}
            for i, aspect in enumerate(tmp['category'].unique()):
                train_aspect_dic[aspect] = i
            # print('Aspect id:', train_aspect_dic.keys())
            aspect_freq = self.get_aspect_freq(tmp,train_aspect_dic)
            # pickle.dump(attribute_ground_truth, f)
            # sentence_ground_truth = self.get_word_id(tmp)
            # pickle.dump(sentence_ground_truth, f)
            # pickle.dump(train_aspect_dic,f)
        return aspect_freq

    # def load_test_data(self,test_aspect_dic):
    #     if os.path.exists(self.data_config['test_data_file_path']) and os.path.getsize(self.data_config['test_data_file_path']) > 0:
    #         f = open(self.data_config['test_data_file_path'],'rb')
    #         attribute_ground_truth = pickle.load(f)
    #         sentence_ground_truth  = pickle.load(f)
    #         test_aspect_dic = pickle.load(f)
    #         f.close()
    #     else:
    #         f = open(self.data_config['test_data_file_path'], 'wb')
    #         tmp = pd.read_csv(self.data_config['test_source_file_path'], index_col=0)
    #         attribute_ground_truth = self.get_aspect_id(tmp,test_aspect_dic)
    #         pickle.dump(attribute_ground_truth, f)
    #         sentence_ground_truth = self.get_word_id(tmp)
    #         pickle.dump(sentence_ground_truth, f)
    #         pickle.dump(test_aspect_dic,f)
    #         f.close()
    #     return attribute_ground_truth, sentence_ground_truth

