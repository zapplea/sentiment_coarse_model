import numpy as np
import gensim
from sklearn.utils import check_array

class DataGenerator:
    def __init__(self,nn_config):
        self.nn_config = nn_config
        # data = [[review, is_padding, attribute labels],...]; its type is list
        self.data = np.array([1,2,3])
        np.random.shuffle(self.data)
        self.table,self.word2id_dic=self.generate_embedding()

    def generate_embedding(self):
        """
        generate table for look up and the dictionary used to convert words in sentence to word id.
        :return: 
        """
        vecfpath = self.nn_config['wordembedding_file_path']
        word_embed = gensim.models.KeyedVectors.load_word2vec_format(vecfpath, binary=False, datatype=np.float32)
        embed_mat = word_embed.syn0
        vocabulary = word_embed.index2word
        embed_mat = check_array(embed_mat, dtype='float32', order='C')
        dictionary = {}
        for i in range(len(vocabulary)):
            dictionary[vocabulary[i]] = i

        return dictionary, embed_mat

    def table_generator(self):
        return self.table

    def data_generator(self,mode,**kwargs):
        """
        
        :param mode: only 'train' or 'test'
        :return: 
        """
        if mode == 'train':
            batch_num=kwargs['batch_num']
            data_temp=self.data[:-self.nn_config['test_data_size']]
            train_data_size = len(data_temp)
            start = batch_num * self.nn_config['batch_size'] % train_data_size
            end = (batch_num * self.nn_config['batch_size'] + self.nn_config['batch_size']) % train_data_size
            if start < end:
                batch = data_temp[start:end]
            elif start >= end:
                batch = data_temp[start:]
                batch.extend(data_temp[0:end])
        elif mode =='test':
            data_temp=self.data[-self.nn_config['test_data_size']:]
            batch = data_temp

        return batch[:,0],batch[:,1],batch[:,2]