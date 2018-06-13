#parameter
### new paramter
'non_attr_prob': probability of non-attribute in coarse grained data. Since LDA didn't give p(non-attribute|D), so we need to give a value manually.
'complement': the same to attribute function
'max_review_length': the same to attribute function
'non_attr_prob': the LDA model will not generate p(non-attribute,), so we need to manually choose a value.
'aspect_prob_threshold': the same to coare attribute function

#Data:
### sentiment labels:
shape = (batch size, attributes number+1, 3).The sentiment label to a review will be converted to sentiment labels of each sentence automatically by the program.
so just need to input sentiment label at review level.

The non-attribute should be considered and its the last attribute. 
batch= [sentiment for sentence1, ...]
sentiment for sentence1 = [sentiment for attribute1, ..., sentiment for non-attribute]
sentiment for attribute1 = [0,0,1]
The first represent neutral, the second represents negative, the third is positive. When a sentiment is true, then the corresponding position will be 1.
If an attribute(including non-attribute) doesn't appear in a sentence, then its sentiment is non-sentiment, which means that the sentiment for attribute_i
 will be [0,0,0]. 

### attributes labels:
The same to the coarse attribute function, and the shape is (batch size, attributes number).
I will add one position in the end of each vector to represent non-attribute by the program, and the shape will be (batch size, attributes number +1). 
This prograss is finished by the program so don't worry.
The p(a|D) will be converted to labels automatically. and p(non-attribute|D) will be automatically added.

### sentences:
The same to the coarse attribute function in current state, but in the future, we would use the version without punctuation because of dependency parsing.

### wordembedding table
The same to coarse attribute function