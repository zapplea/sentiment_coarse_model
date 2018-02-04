"""
Used for training word2vec,and return the word2vec model

"""


import sys
reload(sys)
sys.setdefaultencoding('utf8')

import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

import jieba
import pandas as pd
import sys
sys.setrecursionlimit(1000000)
import pickle

#parameters
vocab_dim = 100
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 10  # ingore word with frequence less than this one
window_size = 7
batch_size = 32

input_length = 100
cpu_count = multiprocessing.cpu_count()


neg=pd.read_excel('neg.xls',header=None,index=None)
pos=pd.read_excel('pos.xls',header=None,index=None)
combined=np.concatenate((pos[0], neg[0]))
y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))

def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text

tok = tokenizer(combined)

model = Word2Vec(size=vocab_dim,
                 min_count=n_exposures,
                 window=window_size,
                 workers=cpu_count,
                 iter=n_iterations)
model.build_vocab(tok)
model.train(tok)

model.save('word2vec_model.pkl')