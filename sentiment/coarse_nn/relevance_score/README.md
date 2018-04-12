## what is new?
multiply p(x|a) and p(a|D) to the attribute score.

## where to implement atr_data_generator:
in util/coarse/

## How to use relevance score
1. pre-processing of data:
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
           
2. In atr_data_generator, use function: data_generator to provide reviews and its labels.
   it should provide data in the following format:
   eg. Reviews = [review_1, review_2, ...., review_n]
       reivew_i = [sentence_1, sentence_2, ...,sentence_l]
       sentence_j = [word_1,...,word_m]
       the data_generator return Reviews. The shape of Reviews is (batch size, max review length, words num(max sentence length), word dim)

3. In atr_data_generator, add function: aspect_prob_generator to provide aspect probability.
   also need to provide probability,from topic model, of the aspect in the review.
   eg. aspect_prob = [review_1_aspect_probability,
                      review_2_aspect_probability,
                      ..,
                      review_n_aspect_probabilty]
       review_i_aspect_probability = [aspect_1_prob, aspect_2_prob, ... aspect_k_prob]
       the data generator return aspect_prob.