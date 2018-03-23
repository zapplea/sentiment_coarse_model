import numpy as np
import pandas as pd
import gensim
from nltk.corpus import stopwords
import string

class DataGenerator():
    def __init__(self,data_config):
        self.data_config = data_config
        self.word_embed = gensim.models.KeyedVectors.load_word2vec_format(self.data_config['wordembedding_file_path'], binary=True, unicode_errors='ignore')
        self.vocabulary = self.word_embed.index2word
        self.dictionary = {}
        for i in range(len(self.vocabulary)):
            self.dictionary[self.vocabulary[i]] = i

        self.data = pd.read_csv(self.data_config['data_file_path'], index_col=0)
        self.aspect_dic = {}
        self.sent_dic = {}
        for i, aspect in enumerate(self.data['category'].unique()):
            self.aspect_dic[aspect] = i
        print('Aspect id:',self.aspect_dic.keys())
        for i, sentiment in enumerate(self.data['polarity'].unique()):
            self.sent_dic[sentiment] = i
        print('Sentiment id:', self.sent_dic.keys())



    def data_generator(self,mode,batch_num):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch. 
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """
        # [( emb_id,fname,row_index m_id,c_id,typeText)]

        if mode == 'train':
            train_size = self.data.shape[0]  - self.data_config['testset_size']
            start = batch_num * self.data_config['batch_size'] % train_size
            end = (batch_num * self.data_config['batch_size'] + self.data_config['batch_size']) % train_size
            if start < end:
                batches_sentiment_ground_truth = self.get_sentiment_id(self.data,start,end)
                batches_attribute_ground_truth = self.get_aspect_id(self.data,start,end)
                batches_sentence_ground_truth = self.get_word_id(self.data,start,end)
            elif start >= end:
                batches_sentiment_ground_truth = self.get_sentiment_id(self.data,train_size-self.data_config['batch_size'],train_size)
                batches_attribute_ground_truth = self.get_aspect_id(self.data,train_size-self.data_config['batch_size'],train_size)
                batches_sentence_ground_truth = self.get_word_id(self.data,train_size-self.data_config['batch_size'],train_size)
        else:
            start =  self.data.shape[0] - self.data_config['testset_size']
            end =  self.data.shape[0]
            batches_sentiment_ground_truth = self.get_sentiment_id(self.data,start, end)
            batches_attribute_ground_truth = self.get_aspect_id(self.data,start,end)
            batches_sentence_ground_truth = self.get_word_id(self.data,start,end)
        # during validation and test, to avoid errors are counted repeatedly,
        # we need to avoid the same data sended back repeately
        return batches_sentence_ground_truth, batches_attribute_ground_truth,batches_sentiment_ground_truth

    def table_generator(self):
        """
        Generate word embedding matirx
        :return: word embeddings table: shape = (number of words, word dimension) word1 [0.223, -4.222, ....] word2 [0.883, 0.333, ....] ... ...
        """
        return self.word_embed.syn0, self.dictionary

    def get_aspect_id(self,data,start,end):
        """
        Generate attribute ground truth
        :param data: 
        :param start: 
        :param end: 
        :return: shape = (batch size, attribute numbers) eg. [[1,0,1,...],[0,0,1,...],...]
        """
        aspect = []
        for i in np.arange(start,end):
            vec = np.zeros(len(self.aspect_dic))
            for j in data[data['sentence_id'] == data.iloc[i]['sentence_id']]['category'].unique():
                    vec[self.aspect_dic[j]] = 1
            aspect.append(vec)
        return np.array(aspect)

    def get_sentiment_id(self,data,start,end):
        """
        Generate sentiment ground truth
        :param data: 
        :param start: 
        :param end: 
        :return:  shape =(batch size, attributes number +1, 3) 
        """
        sentiment = []
        for i in np.arange(start,end):
            vec = []
            for j in self.aspect_dic.keys():
                if j in data[data['sentence_id'] == data.iloc[i]['sentence_id']]['category'].unique() and self.sent_dic[data.iloc[i]['polarity']]<3:
                    tmp = np.zeros(3)
                    tmp[self.sent_dic[data.iloc[i]['polarity']]] = 1
                    vec.append(tmp)
                else:
                    vec.append(np.zeros(3))
            sentiment.append(np.array(vec))
        return np.array(sentiment)

    def get_word_id(self,data,start,end):
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
        stopWords = set(pd.read_table(self.data_config['stopwords_file_path'],header=None,delimiter="\t")[0]) | set(stopwords.words('english'))

        sentence = []
        for i in np.arange(start,end):
            a = []
            num = 0
            for word in data.iloc[i]['sentence'].lower().translate(trantab).split():
                if num >= self.data_config['words_num']:
                    break
                if word not in stopWords and word in self.word_embed.index2word:
                    a.append(self.dictionary[word])
                    num += 1
            if num < self.data_config['words_num']:
                for j in range(self.data_config['words_num'] - num):
                    a.append(self.data_config['padding_word_index'])
            sentence.append(np.array(a))
        return np.array(sentence)

class DataGenerator_random():
    """
    Random DataGenerator for test
    """
    def __init__(self, data_config):
        self.data_config = data_config
        self.word_embed = gensim.models.KeyedVectors.load_word2vec_format(
            self.data_config['wordembedding_file_path'], binary=True,
            unicode_errors='ignore')
        self.vocabulary = self.word_embed.index2word
        self.dictionary = {}
        for i in range(len(self.vocabulary)):
            self.dictionary[self.vocabulary[i]] = i

        self.data = pd.read_csv(self.data_config['data_file_path'], index_col=0)
        self.aspect_dic = {}
        self.sent_dic = {}
        for i, aspect in enumerate(set(self.data['category'])):
            self.aspect_dic[aspect] = i
        for i, sentiment in enumerate(set(self.data['polarity'])):
            self.sent_dic[sentiment] = i

    def data_generator(self, mode, batch_num):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch. 
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """
        # [( emb_id,fname,row_index m_id,c_id,typeText)]

        np.random.seed(batch_num % 2000)

        if mode == 'train':
            batches_sentiment_ground_truth = np.random.randint(low=0,high=2,size=(self.data_config['batch_size'],13,3))
            #batches_attribute_ground_truth = np.random.randint(low=0,high=13,size=(self.data_config['batch_size'],13))                                                            )
            batches_attribute_ground_truth = np.random.randint(low=0,high=14,size=(self.data_config['batch_size'],13))
            batches_sentence_ground_truth = np.random.randint(low=0, high=3000001,size=(self.data_config['batch_size'], self.data_config['words_num']))
        else:
            batches_sentiment_ground_truth = np.random.randint(low=0, high=2,
                                                               size=(1000, 13, 3))
            # batches_attribute_ground_truth = np.random.randint(low=0,high=13,size=(self.data_config['batch_size'],13))                                                            )
            batches_sentence_ground_truth = np.random.randint(low=0, high=3000001,
                                                              size=(1000, self.data_config['words_num']))
            batches_attribute_ground_truth = np.random.randint(low=0, high=14,
                                                               size=(1000,13))
        # during validation and test, to avoid errors are counted repeatedly,
        # we need to avoid the same data sended back repeately
        return batches_sentence_ground_truth, batches_attribute_ground_truth, batches_sentiment_ground_truth
