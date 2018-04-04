import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
else:
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.sep_nn.fine_senti_classifier.classifier import Classifier
from sentiment.util.fine.atr_data_generator import DataGenerator
from sentiment.util.fine.metrics import Metrics

def main(nn_config,data_config):
    dg = DataGenerator(data_config)
    cl = Classifier(nn_config,dg)
    epochs_pred_labels,epochs_loss,true_labels = cl.train()
    # TODO: To feed sentence and attributes to the model, we need to convert words and attributes to ids
    # TODO: need to provide two dictionaries: 1. can convert id of attributes to text
    # TODO:                                   2. can convert id of a word in sentence to text
    # TODO: Write two functions in util/fine/atr_data_generator.py to provide them.
    # TODO: function name: atr_dictionary_generator(); word_dictionary_generator()
    atr_dictioanry = dg.atr_dictionary_generator()
    word_dictionary = dg.word_dictionary_generator()

    for i in range(len(epochs_pred_labels)):
        pred_labels = epochs_pred_labels[i]
        loss = epochs_loss[i]
        metrics_dic = Metrics.measure(true_labels,pred_labels,atr_dictioanry)



if __name__ == "__main__":

    seed = {'lstm_cell_size': 300,
            'word_dim': 200
            }
    nn_configs = [
                {  # fixed parameter
                    'attributes_num': 20,
                    'attribute_dim': seed['lstm_cell_size'],
                    'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                    'words_num': 10,
                    'word_dim': seed['word_dim'],
                    'is_mat': True,
                    'epoch': 10000,
                    'batch_size': 30,
                    'lstm_cell_size': seed['lstm_cell_size'],
                    'lookup_table_words_num': 2981402,  # 2074276 for Chinese word embedding
                    'padding_word_index': 1,  # the index of #PAD# in word embeddings list
                    # flexible parameter
                    'reg_rate': 0.03,
                    'lr': 0.003,  # learing rate
                    'atr_pred_threshold': 0, # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
                },
                {  # fixed parameter
                    'attributes_num': 20,
                    'attribute_dim': seed['word_dim'],
                    'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                    'words_num': 10,
                    'word_dim': seed['word_dim'],
                    'is_mat': True,
                    'epoch': 10000,
                    'batch_size': 30,
                    'lstm_cell_size': seed['lstm_cell_size'],
                    'lookup_table_words_num': 2981402,  # 2074276 for Chinese word embedding
                    'padding_word_index': 1,  # the index of #PAD# in word embeddings list
                    # flexible parameter
                    'reg_rate': 0.03,
                    'lr': 0.003,  # learing rate
                    'atr_pred_threshold': 0, # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
                }
                  ]
    data_config = {}
    for nn_config in nn_configs:
        main(nn_config,data_config)