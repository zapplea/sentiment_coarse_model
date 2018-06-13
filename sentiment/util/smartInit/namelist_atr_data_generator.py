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
        self.train_attribute_ground_truth, self.train_sentence_ground_truth,self.aspect_dic , self.dictionary,self.table ,self.train_namelist_vec = self.load_train_data()
        self.test_attribute_ground_truth, self.test_sentence_ground_truth,self.test_namelist_vec = self.load_test_data(self.aspect_dic,self.dictionary)
        self.train_data_size = len(self.train_attribute_ground_truth)
        self.test_data_size = len(self.test_attribute_ground_truth)


    def train_data_generator(self,batch_num,epoch_num):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch. 
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """
        # [( emb_id,fname,row_index m_id,c_id,typeText)]


        train_size = self.train_data_size
        start = batch_num * self.nn_config['batch_size'] % train_size
        end = (batch_num * self.nn_config['batch_size'] + self.nn_config['batch_size']) % train_size
        if start < end:
            batches_attribute_ground_truth = self.train_attribute_ground_truth[start:end]
            batches_sentence_ground_truth = self.train_sentence_ground_truth[start:end]
            batches_namelist_vec = self.train_namelist_vec[start:end]
        else:
            batches_attribute_ground_truth = self.train_attribute_ground_truth[
                                             train_size - self.nn_config['batch_size']:train_size]
            batches_sentence_ground_truth = self.train_sentence_ground_truth[
                                            train_size - self.nn_config['batch_size']:train_size]
            batches_namelist_vec = self.train_namelist_vec[
                                            train_size - self.nn_config['batch_size']:train_size]
        # during validation and test, to avoid errors are counted repeatedly,
        # we need to avoid the same data sended back repeately
        return batches_sentence_ground_truth, batches_attribute_ground_truth , batches_namelist_vec

    def test_data_generator(self):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch. 
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """
        return self.test_sentence_ground_truth,self.test_attribute_ground_truth ,self.test_namelist_vec

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

    def get_word_id(self,data,dictionary,namelist_dic):
        """
        Generate sentences id matrix
        :param data: 
        :param start: 
        :param end: 
        :return: shape = (batch size, words number) eg. [[1,4,6,8,...],[7,1,5,10,...],]
        """

        outtab = ' ' * len(string.punctuation)
        intab = string.punctuation
        trantab = str.maketrans(intab, outtab)
        sentence = []
        namelist_vec = []
        for i in np.arange(0,data.shape[0]):
            a = []
            b = [0]*self.nn_config['attributes_num']
            num = 0
            for word in data.iloc[i]['sentence'].lower().translate(trantab).split():
                for id , namelist_asp in enumerate(namelist_dic):
                    if word in namelist_dic[namelist_asp]:
                        print(i,word,' in ', namelist_asp)
                        b[id] = 1
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
            namelist_vec.append(np.array(b))
        return np.array(sentence),np.array(namelist_vec)

    def get_word_list(self, data ,dictionary):
        outtab = ' ' * len(string.punctuation)
        intab = string.punctuation
        trantab = str.maketrans(intab, outtab)
        word_list = []
        for i in np.arange(0, data.shape[0]):
            for word in data.iloc[i]['sentence'].lower().translate(trantab).split():
                if word in dictionary.keys():
                    word_list.append(dictionary[word])
        word_list = list(set(word_list))
        print('The number of words in dataset:',len(word_list))
        return word_list

    def smart_initiator(self,aspect_dic , dictionary , model):
        """
        :param attributes: ndarray, shape=(attribute numbers ,2, attribute dim)
        :return: 
        """
        # random_mat.shape = (attributes number, attribute mat size-2, attribute dim)
        punctuation = '_'
        res = []
        word_list = []
        for item in list(aspect_dic.keys()):
            a = item.lower().split('#')
            aspect = a[0].split(punctuation)
            attribute = a[1].split(punctuation)
            word_list += [dictionary[i] for i in aspect if i in dictionary.keys()] + [dictionary[i] for i in attribute if i in dictionary.keys()]
            asp_embed = []
            att_embed = []
            for i in aspect:
                    asp_embed.append(model.wv[i])
            for i in attribute:
                if i == 'general' or i == 'style' or i == 'options':
                    att_embed.append(np.random.normal(0, 1, 300))
                else:
                    att_embed.append(model.wv[i])
            res.append([np.array(asp_embed).mean(axis=0), np.array(att_embed).mean(axis=0)])
        return np.array(res) , word_list

    def load_namelist(self,aspect_dic):
        namelist_dic = {}
        for i in aspect_dic:
            filename = self.data_config['namelist_path'] + i + '_pmi_idf.csv'
            namelist_word = pd.read_csv(filename,header=None)[:20][0].tolist()
            namelist_dic[i] = namelist_word
        return namelist_dic

    def load_train_data(self):
        if os.path.exists(self.data_config['train_data_file_path']) and os.path.getsize(self.data_config['train_data_file_path']) > 0:
            f = open(self.data_config['train_data_file_path'],'rb')
            aspect_dic = pickle.load(f)
            dictionary = pickle.load(f)
            attribute_ground_truth = pickle.load(f)
            sentence_ground_truth  = pickle.load(f)
            train_namelist_vec = pickle.load(f)
            table = pickle.load(f)
            f.close()
        else:
            f = open(self.data_config['train_data_file_path'], 'wb')
            tmp = pd.read_csv(self.data_config['train_source_file_path'], index_col=0)
            word_embed = gensim.models.KeyedVectors.load_word2vec_format(self.data_config['wordembedding_file_path'],binary=True, unicode_errors='ignore')
            vocabulary = word_embed.index2word
            pre_dictionary = {}
            for i in range(len(vocabulary)):
                pre_dictionary[vocabulary[i]] = i

            ##Generate attribute_dic
            aspect_dic = {}
            i = 0
            for aspect in tmp['category'].unique():
                if aspect == aspect:
                    aspect_dic[aspect] = i
                    i += 1
            pickle.dump(aspect_dic,f)

            ###Generate dictionary
            trian_word_list = self.get_word_list(tmp, pre_dictionary)
            test_word_list = self.get_word_list(pd.read_csv(self.data_config['test_source_file_path'], index_col=0), pre_dictionary)
            word_list = list(set(trian_word_list + test_word_list))
            vocabulary = list(np.array(vocabulary)[word_list])
            vocabulary.append('#UNK#')
            vocabulary.append('#PAD#')
            dictionary = {}
            for i in range(len(vocabulary)):
                dictionary[vocabulary[i]] = i
            pickle.dump(dictionary, f)

            attribute_ground_truth = self.get_aspect_id(tmp,aspect_dic)
            train_data_mask = tmp['sentence'].drop_duplicates().index
            attribute_ground_truth = attribute_ground_truth[train_data_mask]
            pickle.dump(attribute_ground_truth, f)
            namelist_dic = self.load_namelist(aspect_dic)
            sentence_ground_truth , train_namelist_vec = self.get_word_id(tmp,dictionary,namelist_dic)
            sentence_ground_truth = sentence_ground_truth[train_data_mask]
            train_namelist_vec = train_namelist_vec[train_data_mask]
            pickle.dump(sentence_ground_truth, f)
            pickle.dump(train_namelist_vec, f)


            ###Generate table
            table = self.table_generator(word_embed,word_list)
            print(table.shape)
            pickle.dump(table, f , protocol = 4)

            f.close()

        return attribute_ground_truth, sentence_ground_truth , aspect_dic , dictionary ,table  , train_namelist_vec

    def load_test_data(self,test_aspect_dic,dictionary):
        if os.path.exists(self.data_config['test_data_file_path']) and os.path.getsize(self.data_config['test_data_file_path']) > 0:
            f = open(self.data_config['test_data_file_path'],'rb')
            attribute_ground_truth = pickle.load(f)
            sentence_ground_truth  = pickle.load(f)
            test_namelist_vec  = pickle.load(f)
            f.close()
        else:
            f = open(self.data_config['test_data_file_path'], 'wb')
            tmp = pd.read_csv(self.data_config['test_source_file_path'], index_col=0)
            test_data_mask = tmp['sentence'].drop_duplicates().index
            attribute_ground_truth = self.get_aspect_id(tmp,test_aspect_dic)
            attribute_ground_truth = attribute_ground_truth[test_data_mask]
            pickle.dump(attribute_ground_truth, f)
            namelist_dic = self.load_namelist(test_aspect_dic)
            sentence_ground_truth, test_namelist_vec = self.get_word_id(tmp, dictionary, namelist_dic)
            sentence_ground_truth = sentence_ground_truth[test_data_mask]
            test_namelist_vec = test_namelist_vec[test_data_mask]
            pickle.dump(sentence_ground_truth, f)
            pickle.dump(test_namelist_vec , f)
            f.close()
        return attribute_ground_truth, sentence_ground_truth , test_namelist_vec




