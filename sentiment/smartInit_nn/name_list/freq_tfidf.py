import pickle
import numpy as np
import gensim


class NameList:
    def __init__(self, wordEmbedding_filtPath):
        self.wordEmbedding_filePath = wordEmbedding_filtPath

    def name_list_generator(self):
        # dic={'AMBIENCE#GENERAL':[['atmosphere','romantic','ambience'],['restaurant','place','food','service','decor','service']],
        #   'DRINKS#PRICES':[['wine','wines','water','drinks'],['prices']],
        #   'DRINKS#QUALITY':[['drinks','wine','sake'],['service','glass','bar']],
        #   'DRINKS#STYLE_OPTIONS':[['wine','wines','bear','drinks'],['list','menu','selection','bar']],
        #   'FOOD#PRICES':[['food'],['price','overprice','prices','worth']],
        #   'FOOD#QUALITY':[['food','pizza','sushi','dishes','chicken'],['amazing','fresh','excellent','delicious','good','great','best']],
        #   'FOOD#STYLE_OPTIONS':[['food','menu','portions','rolls'],['small','even','fresh','large','limited','good','great','huge','delicious','fresh']],
        #   'LOCATION#GENERAL':[['location','place','city'],['view','avenue','river','neighborhood','restaurant']],
        #   'RESTAURANT#GENERAL':[['restaurant','place'],['experience','time','food']],
        #   'RESTAURANT#MISCELLANEOUS':[['restaurant'],['place','spot','friends','time','dinner','food','occasion']],
        #   'RESTAURANT#PRICES':[['restaurant','place',],['prices','price','worth','money','value']],
        #   'SERVICE#GENERAL':[['service'],['food','staff','us','place','waiter','restaurant','table']]
        #   }

        pmi_idf = {
            'AMBIENCE#GENERAL': [['disney', 'louder', 'herb', 'path', 'decorative'],
                                 ['whoever', 'delightfully', 'zero', 'gimmick', 'calm']],
            'DRINKS#PRICES': [['board', 'martinis', 'bottles', 'drinks', 'blasted'],
                              ['values', 'dollar', 'minimun', 'system', 'bev']],
            'DRINKS#QUALITY': [['tequila', 'mojito', 'lassi', 'lassi', 'lassi'],
                               ['guaranteed', 'fav', 'sassy', 'successfully', 'complimented']],
            'DRINKS#STYLE_OPTIONS': [['bubbly', 'tap', 'beers', 'champagne', 'noodles'],
                                     ['brahma', 'bombay', 'bottled', 'filled', 'purple']],
            'FOOD#PRICES': [['amazingly', 'farms', 'guacamole', 'sandwiches', 'tilapia'],
                            ['beaten', 'item', 'argue', 'worth', 'pricing']],
            'FOOD#QUALITY': [['cooking', 'curry', 'wagyu', 'dishes', 'chicken'],
                             ['uncooked', 'fresh', 'slices', 'delicious', 'good']],
            'FOOD#STYLE_OPTIONS': [['cheeseburgers', 'pimentos', 'portions', 'rolls', 'diet'],
                                   ['concept', 'messy', 'sliced', 'overpack', 'fresh']],
            'LOCATION#GENERAL': [['window', 'sidewalk', 'conveniently', 'block', 'lawns'],
                                 ['requesting', 'liberty', 'chart', 'hudson', 'bedford']],
            'RESTAURANT#GENERAL': [['restaurant', 'snacks', 'market', 'pink', 'oasis'],
                                   ['partial', 'month', 'terms', 'th', 'chennai']],
            'RESTAURANT#MISCELLANEOUS': [['restaurant', 'entertained', 'store', 'crowds', 'couples'],
                                         ['code', 'results', 'hmmm', 'shanghai', 'sign']],
            'RESTAURANT#PRICES': [['restaurant', 'place', 'occasions', 'romatic', 'wonderfully'],
                                  ['prices', 'moderate', 'scammers', 'earned', 'charge']],
            'SERVICE#GENERAL': [['service', 'managed', 'regulars', 'disturb', 'cooperative'],
                                ['silverware', 'technique', 'issues', 'concerned', 'young']]
        }
        return pmi_idf

    def wordEmbedding_loader(self):
        """

        :return: a dict : {word: wordEmbedding, ...}
        """
        word_embed = gensim.models.KeyedVectors.load_word2vec_format(self.wordEmbedding_filePath, binary=True,
                                                                     unicode_errors='ignore')
        vocab = word_embed.vocab
        dictinonary = {}
        for item in vocab:
            dictinonary[item] = word_embed.wv[item]
        return dictinonary

    def mapping(self, name_list, wordEmbedding):
        """
        :param name_list: 
        :param wordEmbedding: dict: {word: wordEmbedding, ...} 
        :return: 
        """
        initializer_dic = {}
        for key in name_list:
            initializer_dic[key] = []
            mention_list = name_list[key]
            for mentions in mention_list:
                mention_vec_list = []
                for mention in mentions:
                    if mention in wordEmbedding:
                        mention_vec_list.append(wordEmbedding[mention])
                mention_vec = np.mean(mention_vec_list, axis=0).astype('float32')
                initializer_dic[key].append(mention_vec)
        return initializer_dic


if __name__ == "__main__":
    config = {
        'pkl_filePath': '/home/lujunyu/repository/sentiment_coarse_model/sentiment/smartInit_nn/name_list/pmi_idf.pkl',
        'wordEmbedding_filePath': '/home/lujunyu/dataset/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin'}
    nl = NameList(config['wordEmbedding_filePath'])
    wordEmbedding = nl.wordEmbedding_loader()
    name_list = nl.name_list_generator()
    initilizer = nl.mapping(name_list, wordEmbedding)
    with open(config['pkl_filePath'], 'wb') as f:
        pickle.dump(initilizer, f)