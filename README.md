# sentiment_coarse_model

Paper: uploaded to dropbox

The model to train:
1. (a-o)*e: sentiment_coarse_model/sentiment/sep_nn/coarse_atr_classifier_1w
2. (a-o)*h: sentiment_coarse_model/sentiment/sep_nn/coarse_atr_classifier_nw
3. (a-o)*h+(a-o)*e: sentiment_coarse_model/sentiment/sep_nn/coarse_atr_classifier_1pNw

Notes:
1. Can run unittest af_unittest.py for each model. 
2. Before run the unittest, need to change PATH in: sys.path.append('/home/liu121/dlnlp')
3. For training, the hyper-parameter should be the same to these in unittest, but you can change values.


input data format:
attribute function:
1. ground truth: shape = (batch size, attribute numbers); use binary way to represent whether a attribute is activated
in a sentence.
eg. [[1,0,1,...],[0,0,1,...],...]
2. sentences: shape = (batch size, words number); use a number to represent each word, the number is corresponding to
the index of the word in word embedding tables.
eg. [[1,4,6,8,...],[7,1,5,10,...],]
3. word embeddings table: shape = (number of words, word dimension); this table contain word embeddings.
Its structure would be:
word1 [0.223, -4.222, ....]
word2 [0.883, 0.333, ....]
... ...
The table should also provide a dictionary to map word to index, and this index is input of sentence

sentiment function:
1. sentiment ground truth: shape =(batch size, attributes number +1, 3); The reason we use attributes number +1 is that we need to
consider none-attribute in sentiment function. "3" is the number of polarity: negative, neutral, positive
use binary way to represent whether a attribute's sentiment is activated. For non-attribute, its sentiment should always be neutral.
If a attribute doesn't activated in a sentence, its sentiment should be [0,0,0]

2. attribute ground truth: shape = (batch size, attribute numbers); use binary way to represent whether a attribute is activated
in a sentence. The reason we need to provide attribute ground truth in here is that we assume we know which attribute is activated in the sentence.
eg. [[1,0,1,...],[0,0,1,...],...]

3. sentences: shape = (batch size, words number); use a number to represent each word, the number is corresponding to
the index of the word in word embedding tables.
eg. [[1,4,6,8,...],[7,1,5,10,...],]

4. word embeddings table: shape = (number of words, word dimension); this table contain word embeddings.
Its structure would be:
word1 [0.223, -4.222, ....]
word2 [0.883, 0.333, ....]
... ...
The table should also provide a dictionary to map word to index, and this index is input of sentence


notes:
1. length of sentence is different, so need to pad the sentence to the same length. Check the longest length of a sentence,
and pad all sentence to the same length.
2. batch size is the number of sentences.