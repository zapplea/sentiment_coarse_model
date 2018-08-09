import pickle
import numpy as np
import nltk

class CoarseAtrDataProd:
    def __init__(self,config):
        self.config=config

    def get_aspect_probility(self,data):
        """
        Generate attribute ground truth
        :param data: 
        :param start: 
        :param end: 
        :return: shape = (batch size, attribute numbers) eg. [[1,0,1,...],[0,0,1,...],...]
        """
        aspect = data['Y_att'].tolist()
        return np.array(aspect)

    def get_review_sentiment(self,data):
        star = data['stars'].tolist()
        return np.array(star)

    def get_word_id(self,data,dictionary):
        """
        Generate sentences id matrix
        :param data:
        :param start:
        :param end:
        :return: shape = (batch size, words number) eg. [[1,4,6,8,...],[7,1,5,10,...],]
        """

        review = []
        review_length=[]
        punctuation = '!"#&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        outtab = ' ' * len(punctuation)
        intab = punctuation
        trantab = str.maketrans(intab, outtab)
        for idx in range(data.shape[0]):
            sens = nltk.sent_tokenize(data.iloc[idx]['text'])
            if len(sens) > self.config['max_review_length']:
                sens = sens[:self.config['max_review_length']]
            sentence = []
            for sent in sens:
                a = []
                num = 0
                for word in sent.lower().translate(trantab).split():
                    if num >= self.config['words_num']:
                        break
                    a.append(dictionary[word])
                    num += 1
                if num < self.config['words_num']:
                    for j in range(self.config['words_num'] - num):
                        a.append(self.config['padding_word_index'])
                else:
                    a = a[:self.config['words_num']]

                sentence.append(np.array(a))
            review_length.append(len(sentence))
            if len(sentence) < self.config['max_review_length']:
                for j in range(self.config['max_review_length'] - len(sentence)):
                    sentence.append(np.array([dictionary['#PAD#']] * self.config['words_num']))
            else:
                sentence = sentence[:self.config['max_review_length']]
            review.append(np.array(sentence))
        return np.array(review),np.array(review_length)

    def add_other(self,sentence_len,labels,stars):
        O = np.zeros(shape=(len(labels),))
        O = np.expand_dims(O,axis=1)
        print('labes shape: ',labels.shape)
        print('O shapes: ',O.shape)
        labels = np.concatenate([labels,O],axis=1)
        for i in range(len(sentence_len)):
            length = sentence_len[i]
            label = labels[i]
            star = stars[i]
            if star==3:
                label[-1]=1/(3+length)
        return labels


    def train_data_producer(self):
        with open(self.config['train_source_filePath'], 'rb') as f:
            tmp = pickle.load(f)

        ##Generate attribute_dic
        aspect_dic = {'RESTAURANT': 0, 'SERVICE': 1, 'FOOD': 2
            , 'DRINKS': 3, 'AMBIENCE': 4, 'LOCATION': 5, 'OTHER': 6}

        ###Generate word_dic
        with open(self.config['dictionary'], 'rb') as wd:
            word_list, word_dic, word_embed = pickle.load(wd)
            print('The number of words in dataset:', len(word_dic))

        label = self.get_aspect_probility(tmp)
        stars = self.get_review_sentiment(tmp)
        train_data_mask = tmp['text'].drop_duplicates().reset_index().index
        label = label[train_data_mask]
        stars = stars[train_data_mask]
        sentence,sentence_len = self.get_word_id(tmp, word_dic)
        sentence = sentence[train_data_mask]
        sentence_len = sentence_len[train_data_mask]
        label = self.add_other(sentence_len,label,stars)
        print('train data mask: \n',train_data_mask)
        print('sentence shape: ',sentence.shape)
        print('label shape: ',label.shape)
        print('stars shape: ',stars.shape)
        exit()

        ###Generate word_embed
        with open(self.config['train_data_filePath'], 'wb') as f:
            pickle.dump(aspect_dic,f)
            pickle.dump(word_dic,f)
            pickle.dump(label,f)
            pickle.dump(sentence,f)
            pickle.dump(word_embed,f)
            pickle.dump(stars,f)
            # pickle.dump(aspect_dic, word_dic, label, sentence, word_embed,stars, f)
        return word_dic

    def test_data_producer(self,word_dic):
        with open(self.config['test_source_filePath'], 'rb') as ff:
            tmp = pickle.load(ff)
        test_data_mask = tmp['text'].drop_duplicates().reset_index().index
        label = self.get_aspect_probility(tmp)
        stars =self.get_review_sentiment(tmp)
        label = label[test_data_mask]
        stars = stars[test_data_mask]
        sentence, sentence_len = self.get_word_id(tmp, word_dic)
        sentence = sentence[test_data_mask]
        sentence_len = sentence_len[test_data_mask]
        label = self.add_other(sentence_len, label, stars)
        with open(self.config['test_data_filePath'], 'wb') as f:
            pickle.dump(label, f)
            pickle.dump(sentence, f)
            pickle.dump(stars,f)

if __name__=='__main__':
    data_config={
        'train_source_filePath': '/datastore/liu121/sentidata2/expdata/yelp/yelp_lda_trainset.pkl',
        'train_data_filePath': '/datastore/liu121/sentidata2/expdata/coarse_data/coarse_train_data_v2.pkl',
        'test_source_filePath': '/datastore/liu121/sentidata2/expdata/yelp/yelp_lda_testset.pkl',
        'test_data_filePath': '/datastore/liu121/sentidata2/expdata/coarse_data/coarse_test_data_v2.pkl',
        'dictionary': '/datastore/liu121/sentidata2/expdata/data_dictionary.pkl',
        'max_review_length': 30,
        'words_num': 40,
        'padding_word_index': 34933
    }
    prod = CoarseAtrDataProd(data_config)
    word_dic = prod.train_data_producer()
    prod.test_data_producer(word_dic)
