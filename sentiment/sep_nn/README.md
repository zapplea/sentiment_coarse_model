In this directory, we implement sentiment model separately, and train them separately.

Constraints to Hyper-parameter:
1. rps_num <= words_num

basic attribute function module:
use parameter is_mat to choose attribute mention matrix(True) and attribute vector(False).
1.fine_atr_classifier_1w : use word embedding to scan the sentence
2. fine_atr_classifier_nw: use lstm lifted embedding to scan the sentence
3. fine_atr_classifier_1pNw: use word embedding and lstm lifted embedding together to scan the sentence.

smartInit:
Initiate attribute mention matrix with aspect and attribute.
modules adapt this design
1. fine_atr_1w_smartInit
2. fine_atr_nw_smartInit
3. fine_atr_1pNw_smartInit