Constraints to Hyper-parameter:
1w: attribute_dim = word_embedding dim
nw: attribute dim = lstm cell size
1pNw: attribute dim = lstm cell size = word dim

basic attribute function module:
use parameter is_mat to choose attribute mention matrix(True) and attribute vector(False). (in smartInit_nn, it can only be True)
1.fine_atr_classifier_1w : use word embedding to scan the sentence
2. fine_atr_classifier_nw: use lstm lifted embedding to scan the sentence
3. fine_atr_classifier_1pNw: use word embedding and lstm lifted embedding together to scan the sentence.