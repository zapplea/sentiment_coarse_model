In this directory we implement coarse-grained attribute and sentiment function.

## influence of padded sentence
Since the length of reviews is not the same, therefore we need to pad sentences.
eg. a possible review:
[[wrod1, ...],
 [word2, ...],
 ... ...,
 [#PAD#, ...],
 [#PAD#, ...],
 ... ...]
After look up embedding table, the padded sentence will be like [[0,...,0,...,0],...]
The length of padded sentence is 0, so the result of lstm is [[0,...,0,...,0],...].
The score use mask to set (a-o)*h as "-inf", so, for each attribute, the S(a,x) = -inf, where x is padded sentence. Therefore, the max_false_score, S(a',x), the same.
When calculate the loss, 
    theta-S(a,x)+S(a',x)
    = theta -(-inf)+(-inf)
    = NaN
There is a non-attribute mask, to eliminate influence of false loss for attribute-existing case and calculate loss for non-attribute case.
For attribute-existing case, the mask is like [1,0,0,...0], and, in non-attribute case, it's [1/m, ..., 1/m] where m is attributes number. In each case,
the NaN will be preserved. 

for tensorflow, max([0,NaN]) is equal to 0. So we can conclude that the padded sentences will not have any influence to the model.


## Constraints to Hyper-parameter:
1w: attribute_dim = word_embedding dim

nw: attribute dim = lstm cell size

1pNw: attribute dim = lstm cell size = word dim

*** the batch size can be any size, and it is not necessarily to be a factor of length of test data.
since we change the regularization to reduce_mean(loss)+1/m*regularizer == reduce_mean( loss + regularizer)

## basic attribute function module:
use parameter is_mat to choose attribute mention matrix(True) and attribute vector(False).

1. fine_atr_classifier_1w : use word embedding to scan the sentence

2. fine_atr_classifier_nw: use lstm lifted embedding to scan the sentence

3. fine_atr_classifier_1pNw: use word embedding and lstm lifted embedding together to scan the sentence.

# Parameter
## new parameters:
max_review_length: the maximal number of sentences in a review.

aspect_prob_threshold: since the topic is represented by probability value, this threshold is to determine whether the model contain this topic

## basic parameters:
'attributes_num': number of attributes, need to eliminate non-attribute,

'attribute_dim': dimension of attribute,

'attribute_mat_size': number of attribute mention prototypes in a attribute matrix

'words_num': maximal length of sentence,

'word_dim': word dimension,

'is_mat': whether to use attribute mention matrix,

'epoch': number of epoches,

'batch_size': size of a batch,

'lstm_cell_size': size of lstm cell,

'lookup_table_words_num': number of lookup table,  # 300w for google news, 2074276 for Chinese word embedding

'padding_word_index': the index of #PAD# in word embeddings list

'reg_rate': coefficient of regularizer,

'lr': learing rate

'atr_pred_threshold': if attribute socre is greater than the threshold, then the class is set to 1 in pred_label,

'attribute_loss_theta': in max margin loss, used as bound.