import pickle

class CoarseAtrDataGen:
    def __init__(self,config):
        self.config=config

    def generate(self):
        with open(self.config['coarse_train_source_file'], 'rb') as f:
            tmp = pickle.load(f)

        ##Generate attribute_dic
        aspect_dic = {'RESTAURANT': 0, 'SERVICE': 1, 'FOOD': 2
            , 'DRINKS': 3, 'AMBIENCE': 4, 'LOCATION': 5, 'OTHER': 6}

        ###Generate word_dic
        with open(self.config['dictionary'], 'rb') as wd:
            word_list, word_dic, word_embed = pickle.load(wd)
            print('The number of words in dataset:', len(word_dic))

        label = self.get_aspect_probility(tmp)
        train_data_mask = tmp['text'].drop_duplicates().reset_index().index
        label = label[train_data_mask]
        sentence = self.get_word_id(tmp, word_dic)
        sentence = sentence[train_data_mask]

        ###Generate word_embed

        with open(self.configs['coarse_train_data_file'], 'wb') as f:
            pickle.dump((aspect_dic, word_dic, label, sentence, word_embed), f)