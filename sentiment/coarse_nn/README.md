In this directory we implement coarse-grained attribute and sentiment function.

## Constraints to Hyper-parameter:
1w: attribute_dim = word_embedding dim
nw: attribute dim = lstm cell size
1pNw: attribute dim = lstm cell size = word dim
*** the batch size can be any size, and it is not necessarily to be a factor of length of test data.
## chang regularization
reduce_mean(loss)+1/m*regularizer == reduce_mean( loss + regularizer)

## basic attribute function module:
use parameter is_mat to choose attribute mention matrix(True) and attribute vector(False).
1.fine_atr_classifier_1w : use word embedding to scan the sentence
2. fine_atr_classifier_nw: use lstm lifted embedding to scan the sentence
3. fine_atr_classifier_1pNw: use word embedding and lstm lifted embedding together to scan the sentence.

# Parameter
## new parameters:
max_review_length: the maximal number of sentences in a review.

## basic parameters:

## main change in program:
The aspect should be copied, to fit the size of batch size * max review len.
