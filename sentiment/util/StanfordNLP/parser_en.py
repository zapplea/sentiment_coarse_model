import os
from nltk.parse.stanford import StanfordDependencyParser
import nltk

if __name__ == "__main__":
    os.environ['STANFORD_PARSER'] = '/home/yibing/Documents/csiro/stanfordNLP_zip/stanford-parser-full-2018-02-27/stanford-parser.jar'
    os.environ['STANFORD_MODELS'] = '/home/yibing/Documents/csiro/stanfordNLP_zip/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar'
    cn_parser = StanfordDependencyParser(model_path="/home/yibing/Documents/csiro/stanfordNLP_zip/stanford-english-corenlp-2018-02-27-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz",encoding="gb2312")
    sentence="I find a book , a computer and a basketabll under the bed"
    res = list(cn_parser.parse(sentence.split()))
    for row in res[0].triples():
        print(row)
