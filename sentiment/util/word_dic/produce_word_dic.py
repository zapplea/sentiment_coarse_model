import numpy as np
import pandas as pd
import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
import gensim
from nltk.corpus import stopwords
import string
import os
import pickle
from sentiment.util.configure import config

class Word_dictionary():
    def __init__(self):
        self.coarse_train_data = pd.read_pickle(config['coarse_train_source_file'])
        print(self.coarse_train_data.shape)
        self.coarse_test_data = pd.read_pickle(config['coarse_test_source_file'])
        print(self.coarse_test_data.shape)
        self.fine_train_data = pd.read_pickle(config['fine_train_source_file'])
        print(self.fine_train_data.shape)
        self.fine_test_data = pd.read_pickle(config['fine_test_source_file'])
        print(self.fine_test_data.shape)

    def get_dictionary(self):

        ## generate word dictionary from yelp and semeval dataset
        # tmp = pd.concat([self.coarse_train_data['normalization'],self.coarse_test_data['normalization'],self.fine_train_data['normalization'],self.fine_test_data['normalization']],axis=0)
        tmp = set(self.split_word(self.coarse_train_data)+self.split_word(self.coarse_test_data)+self.split_word(self.fine_train_data)+self.split_word(self.fine_test_data))
        word_list, word_dic = self.get_word_dic(tmp)

        ## generate embedding
        with open(config['wordembedding_file_path'], 'rb') as f:
            id2w, w2id, g_embed = pickle.load(f)
        print(g_embed.shape)
        unk_vec = np.mean(g_embed,axis=0)
        print(unk_vec.shape)
        embed4data = []
        for word in word_list:
            if word in id2w:
                embed4data.append(g_embed[w2id[word]])
            else:
                embed4data.append(unk_vec)

        ## add #PAD#
        word_list.append('#PAD#')
        word_dic['#PAD#'] = len(word_dic)
        embed4data.append(np.array([0.0] * 300))

        embed4data = np.array(embed4data)
        print(embed4data.shape)

        with open(config['dictionary'],'wb') as f:
            pickle.dump((word_list, word_dic, embed4data),f)


    def split_word(self,data):
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
        a = []
        for i in range(data.shape[0]):
            for word in data.iloc[i]['text'].lower().translate(trantab).split():
                a.append(word)
        return list(set(a))


    def get_word_dic(self, data):
        word_list= []
        word_dic = {}
        idx = 0
        for word in data:
            if word not in word_list:
                word_list.append(word)
        for word in word_list:
            word_dic[word] = idx
            idx += 1
        return word_list, word_dic

def main():
    wd = Word_dictionary()
    wd.get_dictionary()


if __name__ == "__main__":
    main()