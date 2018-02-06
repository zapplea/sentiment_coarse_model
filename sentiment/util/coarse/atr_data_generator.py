import numpy as np
import h5py
import random
import gensim
from sklearn.utils import check_array

class DataGenerator:
    def __init__(self,nn_config):
        self.nn_config = nn_config
        # data = [[review, is_padding, attribute labels],...]; its type is list
        data = []
        # h5f = h5py.File('/home/lujunyu/repository/sentiment_coarse_model/sentiment/util/coarse/i5s_Probility_matrix.h5','r')
        # atribute_labels = np.array(h5f['iphone5s'][:])[:, 0:self.nn_config['attributes_num']]
        # h5f.close()
        random.seed(1)
        for i in np.arange(1500):
            review = [[random.randint(0, 2000) for r in np.arange(self.nn_config['words_num'])] for rr in np.arange(self.nn_config['sentences_num'])]

            is_padding = [1]*self.nn_config['sentences_num']

            atribute_labels = np.random.dirichlet(np.ones(self.nn_config['attributes_num']), size=1)


            data_tmp = [review,is_padding,list(atribute_labels[0])]
            #print(data_tmp)
            data.append(data_tmp)
        self.data = data
        np.random.shuffle(self.data)
        self.table, self.word2id_dic = self.generate_embedding()


    def generate_embedding(self):
        """
        generate table for look up and the dictionary used to convert words in sentence to word id.
        :return: 
        """
        vecfpath = self.nn_config['wordembedding_file_path']
        word_embed = gensim.models.KeyedVectors.load_word2vec_format(vecfpath,binary=True,unicode_errors='ignore')
        embed_mat = word_embed.syn0
        unk_vec = embed_mat.mean(0)
        embed_mat = list(embed_mat)
        embed_mat.append(list(unk_vec))
        vocabulary = word_embed.index2word
        vocabulary.append('#UNK#')
        embed_mat = check_array(embed_mat, dtype='float32', order='C')
        dictionary = {}
        print("Generating word embedding...")
        for i in range(len(vocabulary)):
            dictionary[vocabulary[i]] = i
        print("Vocabulary :" ,len(vocabulary))
        return embed_mat , dictionary

    def table_generator(self):
        return np.random.uniform(size=(3000,200)).astype('float32')## word embedding

    def data_generator(self,mode,**kwargs):
        """
        
        :param mode: only 'train' or 'test'
        :return: 
        """
        batch_num = kwargs['batch_num']
        if mode == 'train':
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
            data_temp = self.data[-self.nn_config['test_data_size']:]
            test_data_size = len(data_temp)
            start = batch_num * self.nn_config['batch_size'] % test_data_size
            end = (batch_num * self.nn_config['batch_size'] + self.nn_config['batch_size']) % test_data_size
            if start < end:
                batch = data_temp[start:end]
            elif start >= end:
                batch = data_temp[start:]
                batch.extend(data_temp[0:end])
        batch = np.array(batch)

        return list(batch[:,0]),list(batch[:,1]),list(batch[:,2])