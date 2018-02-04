import yaml
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from sklearn.cross_validation import train_test_split
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import jieba
import pandas as pd
import sys
import pinyin as py
sys.setrecursionlimit(1000000)
import pickle

#parameters
vocab_dim = 100
maxlen = 100
n_iterations = 1  
n_exposures = 10  # ingore word with frequence less than 10
window_size = 7
batch_size = 32
n_epoch = 10
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

tok = py.tokText2tokPinYin(tok,'normal')


model = Word2Vec.load('model/normalPinYin2vec_model.pkl')



def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec,combined
    else:
        print 'No data provided...'



def concatenateVec(vec1,vec2):
    """
    concatenate two numpy array into one
    Used for combine word embedding and pinyin embedding
    """
    n1 = len(vec1)
    n2 = len(vec2)
    concated = np.zeros(n1+n2)
    for i in range(n1):
        concated[i] = vec1[i]
    for j in range(n1,n2):
        concated[j] = vec2[j - n1]
    return concated


## modify this
def get_data(index_dict,word_vectors,combined,y):

    n_symbols = len(index_dict) + 1  # plus words which freqence is lower than 10
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print x_train.shape,y_train.shape
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test




index_dict, word_vectors,sentence_tf_representation = create_dictionaries(model,tok)



n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,sentence_tf_representation,y)

print x_train.shape,y_train.shape


def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print 'Defining a Simple Keras Model...'
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print 'Compiling the Model...'
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print "Train..."
    history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch, validation_data=(x_test, y_test),show_accuracy=True)

    print "Evaluate..."
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('OnlyPinYin_lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('OnlyPinYin_lstm.h5')
    print 'Test score:', score
    return history


history = train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)

with open('OnlyPinYinHistoryDict.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)