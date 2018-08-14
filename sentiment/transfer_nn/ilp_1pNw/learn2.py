import os
import sys
import getpass
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import argparse

from sentiment.transfer_nn.ilp_1pNw.classifier import Classifier
from sentiment.util.coarse.atr_data_generator import DataGenerator as coarse_DataGenerator
from sentiment.util.fine.atr_data_feeder import DataFeeder as fine_DataGenerator

def main(coarse_nn_config, fine_nn_config, coarse_data_config, fine_data_config):
    coarse_dg = coarse_DataGenerator(coarse_data_config)
    fine_dg = fine_DataGenerator(fine_data_config)

    cl = Classifier(coarse_nn_config, fine_nn_config, coarse_dg, fine_dg)
    cl.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0, help='the num of lr and reg_rate')
    parser.add_argument('--dataset', type=str, default='3', help='which dataset will be used')
    parser.add_argument('--train_mod', type=str, default='v1', help='the version of loss function')
    args = parser.parse_args()
    seed = {'lstm_cell_size': 300,
            'word_dim': 300,
            'attribute_mat_size':5
            }
    coarse_nn_config = {  # fixed parameter
        'attributes_num': 6,
        'attribute_dim': seed['word_dim'],
        'attribute_mat_size': seed['attribute_mat_size'],
        # number of attribute mention prototypes in a attribute matrix
        'max_review_length': 1,
        'words_num': 40,
        'word_dim': seed['word_dim'],
        'is_mat': True,
        'epoch': 1000,
        'batch_size': 34,
        'lstm_cell_size': seed['lstm_cell_size'],
        'lookup_table_words_num': 30342,  # 2074276 for Chinese word embedding
        'padding_word_index': 30341,  # the index of #PAD# in word embeddings list
        'unk_word_index': 30340,
        # flexible parameter
        'reg_rate': 0.003,
        'lr': 0.0003,  # learing rate
        'atr_pred_threshold': 0,
        # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
        'attribute_loss_theta': 3.0,
        'aspect_prob_threshold': 0.2,
        'keep_prob_lstm': 0.5,
        'complement': 0,
    }
    fine_nn_config = {
        # fixed parameter
        'attributes_num': 12,
        'attribute_dim': seed['word_dim'],
        'attribute_mat_size': seed['attribute_mat_size'],
        # number of attribute mention prototypes in a attribute matrix
        'words_num': 40,
        'word_dim': seed['word_dim'],
        'is_mat': True,
        'epoch': 1000,
        'batch_size': 10,
        'lstm_cell_size': seed['lstm_cell_size'],
        'lookup_table_words_num': 30342,  # 2074276 for Chinese word embedding
        'padding_word_index': coarse_nn_config['padding_word_index'],  # the index of #PAD# in word embeddings list
        'unk_word_index': coarse_nn_config['unk_word_index'],
        # flexible parameter
        # 'reg_rate': 3E-5,
        # 'lr': 3E-4,
        'atr_pred_threshold': 0,
        # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
        'attribute_loss_theta': 3.0,
        'keep_prob_lstm': 0.5,
    }
    coarse_data_config = {
        'attributes_num': coarse_nn_config['attributes_num'],
        'batch_size': coarse_nn_config['batch_size'],
        'words_num': coarse_nn_config['words_num'],
        'padding_word_index': coarse_nn_config['padding_word_index'],
        'word_dim': seed['word_dim'],
    }
    fine_data_config = {
        'attributes_num': fine_nn_config['attributes_num'],
        'batch_size': fine_nn_config['batch_size'],
        'words_num': fine_nn_config['words_num'],
        'padding_word_index': fine_nn_config['padding_word_index'],
        'word_dim': seed['word_dim'],
        'top_k_data': 30
    }

    reg_rate = [3E-5, ]
    lr = [3E-4, ]

    coarse_nn_config[
        'sr_path'] = '/datastore/liu121/sentidata2/resultdata/coarse_nn/model/ckpt%s_dataset%s_reg%s_lr%s_aspect%s_mat%s/' \
                     % (str(args.train_mod), str(args.dataset), str(reg_rate[args.num]), str(lr[args.num]),
                        str(seed['attributes_num']), str(seed['attribute_mat_size']))

    coarse_data_config['train_source_file_path'] = '/datastore/liu121/sentidata2/expdata/yelp/yelp_lda_trainset.pkl'
    coarse_data_config['test_source_file_path'] = '/datastore/liu121/sentidata2/expdata/yelp/yelp_lda_testset.pkl'
    coarse_data_config[
        'train_data_file_path'] = '/datastore/liu121/sentidata2/expdata/coarse_data/coarse_train_data_v%s.pkl' % str(
        args.dataset)
    coarse_data_config[
        'test_data_file_path'] = '/datastore/liu121/sentidata2/expdata/coarse_data/coarse_test_data_v%s.pkl' % str(
        args.dataset)
    coarse_data_config[
        'wordembedding_file_path'] = '/datastore/liu121/wordEmb/googlenews/GoogleNews-vectors-negative300.bin'
    # coarse_data_config['stopwords_file_path'] = '/datastore/liu121/sentidata2/expdata/stopwords.txt'
    coarse_data_config[
        'fine_sentences_file'] = '/datastore/liu121/sentidata2/expdata/fine_data/fine_sentences_data.pkl'
    # coarse_data_config['dictionary'] = '/datastore/liu121/sentidata2/expdata/data_dictionary.pkl'

    fine_nn_config[
        'sr_path'] = '/datastore/liu121/sentidata2/resultdata/transfer/model/ckpt%s_dataset%s_reg%s_lr%s_aspect%s_mat%s/' \
                     % (str(args.train_mod), str(args.dataset), str(reg_rate[args.num]), str(lr[args.num]),
                        str(seed['attributes_num']), str(seed['attribute_mat_size']))
    fine_data_config[
        'train_source_file_path'] = '/datastore/liu121/sentidata2/expdata/semeval2016/absa_resturant_train.pkl'
    fine_data_config[
        'test_source_file_path'] = '/datastore/liu121/sentidata2/expdata/semeval2016/absa_resturant_test.pkl'
    fine_data_config[
        'train_data_file_path'] = '/datastore/liu121/sentidata2/expdata/fine_data/fine_train_data.pkl'
    fine_data_config[
        'test_data_file_path'] = '/datastore/liu121/sentidata2/expdata/fine_data/fine_test_data.pkl'
    fine_data_config[
        'wordembedding_file_path'] = '/datastore/liu121/wordEmb/googlenews/GoogleNews-vectors-negative300.bin'
    # fine_data_config['stopwords_file_path'] = '/datastore/liu121/sentidata2/expdata/stopwords.txt'
    # fine_data_config['dictionary'] = '/datastore/liu121/sentidata2/expdata/data_dictionary.pkl'

    fine_nn_config['reg_rage'] = reg_rate[args.num]
    fine_nn_config['lr'] = lr[args.num]
    fine_nn_config['tfb_filePath'] = fine_nn_config['sr_path']
    fine_nn_config['coarse_attributes_num'] = coarse_nn_config['attributes_num']


    main(coarse_nn_config, fine_nn_config, coarse_data_config, fine_data_config)