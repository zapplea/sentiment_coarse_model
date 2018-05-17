# input batch size:
there is no limitation to the size of batch. You can change code in sentinn_train.

# Metrics
You can write a new metrics program. 
In the classifier, the final output is prediction result with shape (batch size, attributes number+1, 3).

# Data:
###sentiment labels:
shape = (batch size, attributes number+1, 3). The non-attribute should be considered and its the last attribute. 
batch= [sentiment for sentence1, ...]
sentiment for sentence1 = [sentiment for attribute1, ..., sentiment for non-attribute]
sentiment for attribute1 = [0,0,1]
The first represent neutral, the second represents negative, the third is positive. When a sentiment is true, then the corresponding position will be 1.
If an attribute(including non-attribute) doesn't appear in a sentence, then its sentiment is neutral. 

###attributes labels:
The same to the attribute function, and the shape is (batch size, attributes number). 
I will add one position in the end of each vector to represent non-attribute by the program, and the shape will be (batch size, attributes number +1). 
This prograss is finished by the program so don't worry.

###sentences:
The same to the attribute function in current state, but in the future, we would use the version without punctuation because of dependency parsing.

### wordembedding table
The same to attribute function

#Wordembedding:
In the dependency parsing version, the relationship words in sentiment path should be involved. Need to check how many words is unk in the generated sentence.

#Note:
The attribute is given even in the test. So, if the attributes cannot be recognized correctly, the performance of the sentiment will be influenced.
TODO:for coarse model, there should be something to eliminate the influence of padded sentences' labels.

# path dependency version
dependency relathion words: since these words are special and cannot be find in the original word embedding, we extract all of them and form a vocabulary of 
relation words. The relation words are something like "#case#", or "#amod#". They will be randomly initialized, which means we use np.random.normal to give each 
word a embedding. Then this vocabulary will be attached at the end of the original word embeddings, so we can use one lookup table to process sentences and 
dependency path.
In the program, just need to know the number of relation words.

# paramters:
### new paramters:
'rel_words_num': the number of relation words.














