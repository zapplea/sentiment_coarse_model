import numpy as np
import pandas as pd
import gensim
from nltk.corpus import stopwords
import string
import os
import pickle

class DataGenerator():
    def __init__(self,data_config,nn_config):
        self.data_config = data_config
        self.nn_config = nn_config
        self.train_attribute_ground_truth, self.train_sentiment_ground_truth,self.train_sentence_ground_truth,self.aspect_dic ,self.senti_dic , self.dictionary,self.table = self.load_train_data()
        self.val_attribute_ground_truth, self.val_sentiment_ground_truth , self.val_sentence_ground_truth = self.load_test_data(self.aspect_dic, self.senti_dic ,self.dictionary)
        # self.train_sentence_ground_truth, self.train_attribute_ground_truth = self.unison_shuffled_copies(self.train_sentence_ground_truth, self.train_attribute_ground_truth)
        # self.train_attribute_ground_truth,self.train_sentence_ground_truth = self.train_attribute_ground_truth[:self.data_config['top_k_data']] , self.train_sentence_ground_truth[:self.data_config['top_k_data']]
        self.train_data_size = len(self.train_attribute_ground_truth)
        self.val_data_size = len(self.val_attribute_ground_truth)


    def data_generator(self,batch_num,flag):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch. 
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """
        # [( emb_id,fname,row_index m_id,c_id,typeText)]

        if flag == 'train':
            train_size = self.train_data_size
            start = batch_num * self.nn_config['batch_size'] % train_size
            end = (batch_num * self.nn_config['batch_size'] + self.nn_config['batch_size']) % train_size
            if start < end:
                batches_attribute_ground_truth = self.train_attribute_ground_truth[start:end]
                batches_sentiment_ground_truth = self.train_sentiment_ground_truth[start:end]
                batches_sentence_ground_truth = self.train_sentence_ground_truth[start:end]
            else:
                batches_attribute_ground_truth = self.train_attribute_ground_truth[
                                                 train_size - self.nn_config['batch_size']:train_size]
                batches_sentiment_ground_truth = self.train_sentiment_ground_truth[
                                                 train_size - self.nn_config['batch_size']:train_size]
                batches_sentence_ground_truth = self.train_sentence_ground_truth[
                                                train_size - self.nn_config['batch_size']:train_size]
        else:
            val_size = self.val_data_size
            start = batch_num * self.nn_config['batch_size'] % val_size
            end = (batch_num * self.nn_config['batch_size'] + self.nn_config['batch_size']) % val_size
            if start < end:
                batches_attribute_ground_truth = self.train_attribute_ground_truth[start:end]
                batches_sentiment_ground_truth = self.train_sentiment_ground_truth[start:end]
                batches_sentence_ground_truth = self.train_sentence_ground_truth[start:end]
            else:
                batches_attribute_ground_truth = self.train_attribute_ground_truth[
                                                 val_size - self.nn_config['batch_size']:val_size]
                batches_sentiment_ground_truth = self.train_sentiment_ground_truth[
                                                 val_size - self.nn_config['batch_size']:val_size]
                batches_sentence_ground_truth = self.train_sentence_ground_truth[
                                                val_size - self.nn_config['batch_size']:val_size]
        # during validation and test, to avoid errors are counted repeatedly,
        # we need to avoid the same data sended back repeately
        return batches_sentence_ground_truth, batches_attribute_ground_truth ,batches_sentiment_ground_truth

    def test_data_generator(self):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch. 
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """
        return self.test_sentence_ground_truth,self.test_attribute_ground_truth

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
        unk_vec = np.mean(tmp,axis=0).reshape(1,self.nn_config['word_dim'])
        pad_vec = np.zeros((1,self.nn_config['word_dim']))
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
            vec = np.zeros(len(aspect_dic)-1)
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
                    vec.append(np.array([0,0,0]))
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
        for i in np.arange(0,data.shape[0]):
            a = []
            num = 0
            for word in data.iloc[i]['text'].lower().translate(trantab).split():
                if num >= self.nn_config['words_num']:
                    break
                if word in dictionary.keys():
                    a.append(dictionary[word])
                else:
                    a.append(dictionary['#UNK#'])
                num += 1
            if num < self.nn_config['words_num']:
                for j in range(self.nn_config['words_num'] - num):
                    a.append(self.nn_config['padding_word_index'])
            sentence.append(np.array(a))
        return np.array(sentence)

    def get_word_list(self, data ,dictionary):
        outtab = ' ' * len(string.punctuation)
        intab = string.punctuation
        trantab = str.maketrans(intab, outtab)
        word_list = []
        for i in np.arange(0, data.shape[0]):
            for word in data.iloc[i]['text'].lower().translate(trantab).split():
                if word in dictionary.keys():
                    word_list.append(dictionary[word])
        word_list = list(set(word_list))
        print('The number of words in dataset:',len(word_list))
        return word_list


    def load_train_data(self):
        if os.path.exists(self.data_config['train_data_file_path']) and os.path.getsize(self.data_config['train_data_file_path']) > 0:
            f = open(self.data_config['train_data_file_path'],'rb')
            aspect_dic = pickle.load(f)
            senti_dic = pickle.load(f)
            dictionary = pickle.load(f)
            attribute_ground_truth = pickle.load(f)
            sentiment_ground_truth = pickle.load(f)
            sentence_ground_truth  = pickle.load(f)
            table = pickle.load(f)
            f.close()
        else:
            f = open(self.data_config['train_data_file_path'], 'wb')
            tmp = pd.read_pickle(self.data_config['train_source_file_path'])
            word_embed = gensim.models.KeyedVectors.load_word2vec_format(self.data_config['wordembedding_file_path'],binary=True, unicode_errors='ignore')

            ##Generate attribute_dic
            aspect_dic = {}
            i = 0
            for aspect in tmp['category'].unique():
                if aspect == aspect:
                    aspect_dic[aspect] = i
                    i += 1
            aspect_dic['NONE'] = len(aspect_dic)
            pickle.dump(aspect_dic,f)

            ##Generate sentiment_dic
            senti_dic = {'neutral': 0, 'negative': 1, 'positive': 2}
            pickle.dump(senti_dic, f)


            ###Generate dictionary
            with open(self.data_config['dictionary'],'rb') as wd:
                word_list = pickle.load(wd)
                dictionary = pickle.load(wd)
                print('The number of words in dataset:',len(dictionary))
            pickle.dump(dictionary, f)

            attribute_ground_truth = self.get_aspect_id(tmp,aspect_dic)
            train_data_mask = tmp['sentence_id'].drop_duplicates().index
            attribute_ground_truth = attribute_ground_truth[train_data_mask]
            pickle.dump(attribute_ground_truth, f)

            sentiment_ground_truth = self.get_sentiment_id(tmp, aspect_dic,senti_dic)
            sentiment_ground_truth = sentiment_ground_truth[train_data_mask]
            pickle.dump(sentiment_ground_truth, f)

            sentence_ground_truth = self.get_word_id(tmp,dictionary)
            sentence_ground_truth = sentence_ground_truth[train_data_mask]
            pickle.dump(sentence_ground_truth, f)


            ###Generate table
            table = self.table_generator(word_embed,word_list)
            print(table.shape)
            pickle.dump(table, f , protocol = 4)

            f.close()

        return attribute_ground_truth, sentiment_ground_truth , sentence_ground_truth , aspect_dic , senti_dic,dictionary ,table

    def load_test_data(self,test_aspect_dic , test_senti_dic ,dictionary):
        if os.path.exists(self.data_config['test_data_file_path']) and os.path.getsize(self.data_config['test_data_file_path']) > 0:
            f = open(self.data_config['test_data_file_path'],'rb')
            attribute_ground_truth = pickle.load(f)
            sentiment_ground_truth = pickle.load(f)
            sentence_ground_truth  = pickle.load(f)
            f.close()
        else:
            f = open(self.data_config['test_data_file_path'], 'wb')
            tmp = pd.read_pickle(self.data_config['test_source_file_path'])
            test_data_mask = tmp['sentence_id'].drop_duplicates().index
            attribute_ground_truth = self.get_aspect_id(tmp,test_aspect_dic)
            attribute_ground_truth = attribute_ground_truth[test_data_mask]
            pickle.dump(attribute_ground_truth, f)

            sentiment_ground_truth = self.get_sentiment_id(tmp, test_aspect_dic, test_senti_dic)
            sentiment_ground_truth = sentiment_ground_truth[test_data_mask]
            pickle.dump(sentiment_ground_truth, f)

            sentence_ground_truth = self.get_word_id(tmp,dictionary)
            sentence_ground_truth = sentence_ground_truth[test_data_mask]
            pickle.dump(sentence_ground_truth, f)
            f.close()
        return attribute_ground_truth, sentiment_ground_truth, sentence_ground_truth




