## what is new?
The program use window with different size, named as multi-filter, to scan the sentence.
eg.
x = [I, went, to, ANU] 
window size=3
then, after the filter scan through x, it will generate a new sentence S: [[#PAD#, I, went],[I,went,to],[went,to,ANU],[to,ANU,#pad#]].
then we use a feed forward neural network to convert word embedding of each item in S to one word embedding, name as "ei".
Then S will be [e0,e1,e2,e3].
we use S to replace x to do other operation which is the same to sep_nn.

For each filter, we give a different attribute mention matrix (A_mat) or attribute vector (A_vec). 

different generator will produce different score for each attribute.
eg. 
x = [I, went, to, ANU] 
window size=3
window size=5
we get a score for attribute 'a': [1,2,5,0] for windows 3, and [9,0,4,1] for windiows 5
then we concat them to score = [[1,9],[2,0],[5,4],[0,1]]
Then choose the max value for each word, then one attribute's score for x will be [9,2,5,1].
The other thing is the same to sep_nn.

## how to eliminate influence of #PAD# to convolution of multi-filter.
after convolution, the input X's shape will be (batch size, max words num, lstm cell size/word dim). Then the mask in score function will eliminate its influence.

## Changes in use data generator
Nothing has changed compared with data generator to models in sep_nn, but need to copy the sep_nn's data generator to sentiment/util/multifilter/.

## Which data set does we use?
semeval2016 task5