In this directory, we implement sentiment model separately, and train them separately.

## Constraints to Hyper-parameter:
1w, nw, 1pNw: attribute dim = lstm cell size = word dim

batch can be any size.

## basic attribute function module:
use parameter is_mat to choose attribute mention matrix(True) and attribute vector(False).
1.fine_atr_classifier_1w : use word embedding to scan the sentence
2. fine_atr_classifier_nw: use lstm lifted embedding to scan the sentence
3. fine_atr_classifier_1pNw: use word embedding and lstm lifted embedding together to scan the sentence.

# Parameters
## new parameters
filter_size: the size of filter, given by a list, like [1,3,5]. The size of each filter must be a odd number. 
conv_layer_dim:[seed['lstm_cell_size'/'word_dim']]

## basic paramters
'attributes_num': number of attributes, need to eliminate non-attribute

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