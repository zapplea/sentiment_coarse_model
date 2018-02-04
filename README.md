# sentiment_coarse_model

Paper: uploaded to dropbox


TODO_list:
1. Need to complete sentiment_coarse_model/sentiment/util/coarse/atr_data_generator.py. it lacks the part of generating training data, but I provide function data_generator to feed data to training process. You need to use words2id dictionary to convert words in sentences to id at first, and also, need to pad the sentence to the same length(maximum length). The padded word denoted by '#PAD#' and the corrsponding vector is a zeros-vector.
2. Need to write a python file to call classifier and train it. Still, remember to add sys.path.append as in the af_unittest.py


The model to train:
1. (a-o)*e: sentiment_coarse_model/sentiment/sep_nn/coarse_atr_classifier_1w
2. (a-o)*h: sentiment_coarse_model/sentiment/sep_nn/coarse_atr_classifier_nw
3. (a-o)*h+(a-o)*e: sentiment_coarse_model/sentiment/sep_nn/coarse_atr_classifier_1pNw

Notes:
1. Can run unittest af_unittest.py for each model. 
2. Before run the unittest, need to change PATH in: sys.path.append('/home/liu121/dlnlp')
3. For training, the hyper-parameter should be the same to these in unittest, but you can change values.

input data instance:
data=[[review, is_padding, attribute labels],...]
1. review = [sentence_1, sentence_2 ,...,sentence_n]. sentence_i=[word_1,word_2,...,word_m]. Type of word_j is int, representing the code of a word and the reason is that we need to feed the review to lookup_table to extract embeddings. After the word is inputed to lookup_table, it is converted to word embeddings.
2. is_padding is one_hot vector. Each scalar represents whether a sentence is padded, if it is, then the corresponding scalar is 0, otherwise it is 1.
3.attribute_labels is vector of probability. Each scalar respresents possibility of this attributes represented in the review. 

FIX_list:
1. padded word also need a mask.
