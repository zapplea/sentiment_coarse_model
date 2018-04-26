import json
import numpy as np

class NameList:
    def __init__(self,wordEmbedding_filtPath):
          self.wordEmbedding_filePath=wordEmbedding_filtPath

    def name_list_generator(self):
        dic={'AMBIENCE#GENERAL':[['atmosphere','romantic','ambience'],['restaurant','place','food','service','decor','service']],
          'DRINKS#PRICES':[['wine','wines','water','drinks'],['prices']],
          'DRINKS#QUALITY':[['drinks','wine','sake'],['service','glass','bar']],
          'DRINKS#STYLE_OPTIONS':[['wine','wines','bear','drinks'],['list','menu','selection','bar']],
          'FOOD#PRICES':[['food'],['price','overprice','prices','worth']],
          'FOOD#QUALITY':[['food','pizza','sushi','dishes','chicken'],['amazing','fresh','excellent','delicious','good','great','best']],
          'FOOD#STYLE_OPTIONS':[['food','menu','portions','rolls'],['small','even','fresh','large','limited','good','great','huge','delicious','fresh']],
          'LOCATION#GENERAL':[['location','place','city'],['view','avenue','river','neighborhood','restaurant']],
          'RESTAURANT#GENERAL':[['restaurant','place'],['experience','time','food']],
          'RESTAURANT#MISCELLANEOUS':[['restaurant'],['place','spot','friends','time','dinner','food','occasion']],
          'RESTAURANT#PRICES':[['restaurant','place',],['prices','price','worth','money','value']],
          'SERVICE#GENERAL':[['service'],['food','staff','us','place','waiter','restaurant','table']]
          }
        return dic

    def wordEmbedding_loader(self):
        """
        
        :return: a dict : {word: wordEmbedding, ...}
        """
        pass

    def mapping(self,name_list,wordEmbedding):
        """
        :param name_list: 
        :param wordEmbedding: dict: {word: wordEmbedding, ...} 
        :return: 
        """
        initializer_dic={}
        for key in name_list:
            initializer_dic[key]=[]
            mention_list = name_list[key]
            for mentions in mention_list:
                mention_vec_list=[]
                for mention in mentions:
                    if mention in wordEmbedding:
                        mention_vec_list.append(wordEmbedding[mention])
                mention_vec=np.mean(mention_vec_list).astype('float32')
                initializer_dic[key].append(mention_vec)
        return initializer_dic

if __name__=="__main__":
    wordEmbedding_filePath=''
    nl = NameList(wordEmbedding_filePath)
    wordEmbedding = nl.wordEmbedding_loader()
    name_list = nl.name_list_generator()
    initilizer = nl.mapping(name_list,wordEmbedding)
