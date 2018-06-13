import numpy as np
import pandas as pd
import gensim
from nltk.corpus import stopwords
import string
import os
import pickle

class Word_dictionary():
    def __init__(self,coarse_data_path,fine_data_path,wordembed_path):
        self.coarse_data_path = coarse_data_path
        self.fine_data_path = fine_data_path
        self.word_embed = gensim.models.KeyedVectors.load_word2vec_format(wordembed_path, binary=True,
                                                                     unicode_errors='ignore')
        self.vocabulary = self.word_embed.index2word
        self.pre_dictionary = {}
        for i in range(len(self.vocabulary)):
            self.pre_dictionary[self.vocabulary[i]] = i

    def get_dictionary(self):

        coarse_train_data = pd.read_pickle(self.coarse_data_path['train'])
        print(coarse_train_data.shape)
        coarse_test_data = pd.read_pickle(self.coarse_data_path['test'])
        print(coarse_test_data.shape)
        fine_train_data = pd.read_pickle(self.fine_data_path['train'])
        print(fine_train_data.shape)
        fine_test_data = pd.read_pickle(self.fine_data_path['test'])
        print(fine_test_data.shape)

        wordlist = self.get_word_list(coarse_train_data) + self.get_word_list(coarse_test_data) + self.get_word_list(fine_train_data) + self.get_word_list(fine_test_data)
        wordlist = list(set(wordlist))

        vocabulary = list(np.array(self.vocabulary)[wordlist])
        vocabulary.append('#UNK#')
        vocabulary.append('#PAD#')
        print('The number of words in dataset:',len(vocabulary))
        dictionary = {}
        for i in range(len(vocabulary)):
            dictionary[vocabulary[i]] = i
        with open('./data_dictionary.pkl','wb') as f:
            pickle.dump(wordlist,f)
            pickle.dump(dictionary,f)


    def get_word_list(self, data):
        outtab = ' ' * len(string.punctuation)
        intab = string.punctuation
        trantab = str.maketrans(intab, outtab)
        word_list = []
        for i in np.arange(0, data.shape[0]):
            for word in data.iloc[i]['text'].lower().translate(trantab).split():
                if word in self.pre_dictionary.keys():
                    word_list.append(self.pre_dictionary[word])
        return word_list

def main(coarse_data_path,fine_data_path,wordembed_path):
    wd = Word_dictionary(coarse_data_path,fine_data_path,wordembed_path)
    wd.get_dictionary()


if __name__ == "__main__":
    coarse_data_path = {'train':'/home/lujunyu/dataset/yelp/yelp_lda_trainset.pkl',
                      'test':'/home/lujunyu/dataset/yelp/yelp_lda_testset.pkl'}
    fine_data_path = {'train':'/home/lujunyu/dataset/semeval2016/absa_resturant_train.pkl',
                      'test':'/home/lujunyu/dataset/semeval2016/absa_resturant_test.pkl'}
    wordembed_path = '~/dataset/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin'

    main(coarse_data_path,fine_data_path,wordembed_path)