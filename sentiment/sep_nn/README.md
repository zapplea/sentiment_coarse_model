In this directory, we implement sentiment model separately, and train them separately.

## Constraints to Hyper-parameter:
senti function. rps_num <= words_num
1w: attribute_dim = word_embedding dim
nw: attribute dim = lstm cell size
1pNw: attribute dim = lstm cell size = word dim
batch size should be factor of length of test data, otherwise, some test data will be wasted.

## basic attribute function module:
use parameter is_mat to choose attribute mention matrix(True) and attribute vector(False).
1.fine_atr_classifier_1w : use word embedding to scan the sentence
2. fine_atr_classifier_nw: use lstm lifted embedding to scan the sentence
3. fine_atr_classifier_1pNw: use word embedding and lstm lifted embedding together to scan the sentence.

# Parameters
'attributes_num': 13,
'attribute_dim': seed['word_dim'],
'attribute_mat_size': number of attribute mention prototypes in a attribute matrix
'words_num': 20,
'word_dim': seed['word_dim'],
'is_mat': True,
'epoch': 10,
'batch_size': 40,
'lstm_cell_size': seed['lstm_cell_size'],
'lookup_table_words_num': 30000000,  # 2074276 for Chinese word embedding
'padding_word_index': 0,  # the index of #PAD# in word embeddings list
'reg_rate': 0.03,
'lr': 0.03,  # learing rate
'atr_pred_threshold': 0,
'attribute_loss_theta': 1.0,