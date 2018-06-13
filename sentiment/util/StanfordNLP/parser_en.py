import os
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize.stanford import StanfordTokenizer

if __name__ == "__main__":
    os.environ['STANFORD_PARSER'] = '/home/yibing/Documents/csiro/stanfordNLP_zip/stanford-parser-full-2018-02-27/stanford-parser.jar'
    os.environ['STANFORD_MODELS'] = '/home/yibing/Documents/csiro/stanfordNLP_zip/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar'
    en_parser = StanfordDependencyParser(model_path="/home/yibing/Documents/csiro/stanfordNLP_zip/stanford-english-corenlp-2018-02-27-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz",encoding="gb2312")
    sentence="Saul is the best restaurant on Smith Street and in Brooklyn."

    tokenizer = StanfordTokenizer(path_to_jar='/home/yibing/Documents/csiro/stanfordNLP_zip/stanford-parser-full-2018-02-27/stanford-parser.jar')

    res = list(en_parser.parse(tokenizer.tokenize(sentence)))
    for row in res[0].triples():
        print(row)
