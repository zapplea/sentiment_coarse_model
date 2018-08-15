import pickle
import numpy as np

class FineAtrDataProd:
    def __init__(self,config):
        self.config = config

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
        for i in range(data.shape[0]):
            a = []
            num = 0
            for word in data.iloc[i]['text'].lower().translate(trantab).split():
                if num >= self.config['words_num']:
                    break
                a.append(dictionary[word])
                num += 1
            if num < self.config['words_num']:
                for j in range(self.config['words_num'] - num):
                    a.append(self.config['padding_word_index'])
            sentence.append(np.array(a))
        return np.array(sentence)

    def train_data_producer(self):
        with open(self.config['fine_train_source_filePath'], 'rb') as f:
            tmp = pickle.load(f)

        ##Generate attribute_dic
        attribute_dic = {}
        i = 0
        for attribute in tmp['category'].unique():
            if attribute == attribute:
                attribute_dic[attribute] = i
                i += 1

        ###Generate dictionary
        with open(self.config['dictionary'], 'rb') as wd:
            word_list, word_dic, word_embed = pickle.load(wd)
            print('The number of words in dataset:', len(word_dic))

        label = self.get_aspect_id(tmp, attribute_dic)
        train_data_mask = tmp['sentence_id'].drop_duplicates().index
        label = label[train_data_mask]
        sentence = self.get_word_id(tmp, word_dic)
        sentence = sentence[train_data_mask]

        # with open(self.config['fine_train_data_filePath'], 'wb') as f:
        #     pickle.dump((attribute_dic, word_dic, label, sentence, word_embed), f)
        return attribute_dic,word_dic,label,sentence

    def test_data_producer(self,attribute_dic,dictionary):
        with open(self.config['fine_test_source_filePath'], 'rb') as f:
            tmp = pickle.load(f)
        test_data_mask = tmp['sentence_id'].drop_duplicates().index
        label = self.get_aspect_id(tmp, attribute_dic)
        label = label[test_data_mask]
        sentence = self.get_word_id(tmp, dictionary)
        sentence = sentence[test_data_mask]

        # with open(self.config['fine_test_data_filePath'], 'wb') as f:
        #     pickle.dump((label, sentence), f)

    def transfer_data_producer(self,label,sentence):
        fine_sent = []
        for atr_vec in label.transpose():
            r = np.repeat(np.reshape(atr_vec, [2000, 1]), axis=1, repeats=40) * sentence
            r = np.reshape(r[~np.all(r == 0, axis=1)].astype(int),[-1,1,40])
            fine_sent.append(r)
        fine_sent = np.array(fine_sent)
        with open(self.config['fine_sentences_file'], 'wb') as f:
            pickle.dump(fine_sent,f)

if __name__=="__main__":
    config={'words_num':40,
            'padding_word_index':34933,
            'dictionary':'/datastore/liu121/sentidata2/expdata/data_dictionary.pkl',
            'fine_train_source_filePath':'/datastore/liu121/sentidata2/expdata/semeval2016/absa_resturant_train.pkl',
            'fine_train_data_filePath':'/datastore/liu121/sentidata2/expdata/fine_data/fine_train_data.pkl',
            'fine_test_source_filePath':'/datastore/liu121/sentidata2/expdata/semeval2016/absa_resturant_test.pkl',
            'fine_test_data_filePath':'/datastore/liu121/sentidata2/expdata/fine_data/fine_test_data.pkl',
            'fine_sentences_file':'/datastore/liu121/sentidata2/expdata/fine_data/fine_sentences_data.pkl'}
    prod = FineAtrDataProd(config)
    attribute_dic,dictionary,label,sentence=prod.train_data_producer()
    # prod.test_data_producer(attribute_dic,dictionary)
    prod.transfer_data_producer(label,sentence)