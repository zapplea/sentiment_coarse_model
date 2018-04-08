from nltk.tokenize.stanford import StanfordTokenizer

class Tokenizer:
    def __init__(self,path_to_jar='/datastore/liu121/stanfordNLP/stanford_parser/stanford-parser.jar'):
        self.path_to_jar=path_to_jar

    def tokenize(self,sentence):
        tokenizer=StanfordTokenizer(path_to_jar=self.path_to_jar)
        tokens = tokenizer.tokenize(sentence)
        return tokens

    def tokenize_sents(self,sentences):
        tokenizer=StanfordTokenizer(path_to_jar=self.path_to_jar)
        tokens = tokenizer.tokenize_sents(sentences)
        return tokens

if __name__ == "__main__":
    tokenizer = StanfordTokenizer(path_to_jar='/home/yibing/Documents/csiro/stanfordNLP_zip/stanford-parser-full-2018-02-27/stanford-parser.jar')
    sent = ["Leg and toe cramping, leg pain, fasciculations (muscle twitches),"," muscle atrophy, memory problems, peripheral neuropathy . . ."]
    print(tokenizer.tokenize('Judging from previous posts this used to be a good place, but not any longer.'))
    print('========')