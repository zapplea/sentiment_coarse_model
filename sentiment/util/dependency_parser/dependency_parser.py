import os
import sys
sys.path.append('/datastore/liu121/py-package/jieba')

from nltk.parse.stanford import StanfordDependencyParser
import jieba
import nltk

class Node:
    def __int__(self):
        self.key='__empty__'
        # {child:relation}
        self.children={}
        self.parent={}

class DependencyParser:
    def __init__(self,lang):
        self.lang = lang
        os.environ['STANFORD_PARSER'] = '/datastore/liu121/stanfordNLP/stanford_parser/stanford-parser.jar'
        os.environ['STANFORD_MODELS'] = '/datastore/liu121/stanfordNLP/stanford_parser/stanford-parser-3.9.1-models.jar'
        if self.lang == "cn":
            model_path = '/datastore/liu121/stanfordNLP/stanford_cn_model/edu/stanford/nlp/models' \
                         '/lexparser/chinesePCFG.ser.gz'
            self.parser = StanfordDependencyParser(model_path=model_path,encoding="gb2312")
        else:
            model_path = '/datastore/liu121/stanfordNLP/stanford_en_model/edu/stanford/nlp/models' \
                         '/lexparser/englishPCFG.ser.gz'
            self.parser = StanfordDependencyParser(model_path=model_path)

    def parse_cn(self,sentence):
        """
        
        :param tokens: the sentence must be tokens
        :return: 
        """
        tokens = list(jieba.cut(sentence))
        return list(self.parser.parse(tokens))

    def parse_en(self,sentence):
        tokens = nltk.word_tokenize(sentence)
        return list(self.parser.parse(tokens))

    def dependency_graph_builder(self):
        pass