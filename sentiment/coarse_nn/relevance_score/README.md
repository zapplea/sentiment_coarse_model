## what is new?
multiply p(x|a) and p(a|D) to the attribute score.

## How to use relevance score
1. use max review length to control multiplication between attirbute/sentiment function score and relevance probability.
2. pre-processing of data:
   (1) pad each sentence the maximal length;
       eg. sentence = [word1, word2, ..., wordi, #PAD#, ..., #PAD#]
           len(sentence) == words_num (it means maximal sentence length)
   (2) pad the review to the maximal number of sentences.
       eg. Review = [[word1, word2, ..., wordi, #PAD#, ..., #PAD#],
                     [word1, word2, ..., wordi, #PAD#, ..., #PAD#],
                     ...,
                     [word1, word2, ..., wordi, #PAD#, ..., #PAD#],
                     [#PAD#, ..., #PAD#],
                     ...,
                     [#PAD#, ..., #PAD#]]
           len(Review) == max_sentences_num
           
3. the data_generator should provide data in the following format:
   eg. Reviews = [review_1, review_2, ...., review_n]
       reivew_i = [sentence_1, sentence_2, ...,sentence_l]
       sentence_j = [word_1,...,word_m]
       the data_generator return Reviews. The shape of Reviews is (batch size, max review length, words num(max sentence length), word dim)