In this directory we implement coarse-grained attribute and sentiment function.

## Constraints to Hyper-parameter:
1w: attribute_dim = word_embedding dim
nw: attribute dim = lstm cell size
1pNw: attribute dim = lstm cell size = word dim
batch size should be factor of length of test data, otherwise, some test data will be wasted.

## basic attribute function module:
use parameter is_mat to choose attribute mention matrix(True) and attribute vector(False).
1.fine_atr_classifier_1w : use word embedding to scan the sentence
2. fine_atr_classifier_nw: use lstm lifted embedding to scan the sentence
3. fine_atr_classifier_1pNw: use word embedding and lstm lifted embedding together to scan the sentence.

##new parameters:
max_sentences_num: the maximal number of sentences in a review.

