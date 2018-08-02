import getpass
import sys
if getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import pickle
import argparse

from sentiment.transfer_nn.ilp_1pNw.classifier import Classifier

# TODO: analyze init_data
# TODO: 1. mapping between labels
# TODO: 2. nearest neighbour of attribute mention vector
# TODO: check whether coarse and fine use the same table
class Analysis:
    def __init__(self, coarse_nn_config, fine_nn_config, coarse_data_generator, fine_data_generator):
        cl = Classifier(coarse_nn_config, fine_nn_config, coarse_data_generator, fine_data_generator)
        self.coarse_nn_config = coarse_nn_config
        self.fine_nn_config = fine_nn_config
        self.coarse_data_generator = coarse_data_generator
        self.fine_data_generator = fine_data_generator

        with open(self.coarse_data_generator['train_data_file_path'],'rb') as f:
            self.aspect_dic = pickle.load(f)
            self.coarse_word_dic=pickle.load(f)
            self.aspect_labels = pickle.load(f)
            self.coarse_sentences = pickle.load(f)
            self.coarse_table = pickle.load(f)

        with open(self.fine_data_generator['train_data_file_path'],'rb') as f:
            self.attribute_dic = pickle.load(f)
            self.fine_word_dic = pickle.load(f)
            self.attribute_labels=pickle.load(f)
            self.fine_sentences = pickle.load(f)
            self.fine_table = pickle.load(f)

    def check_table(self):
        for word in self.coarse_word_dic:
            coarse_id = self.coarse_word_dic[word]
            fine_id = self.fine_word_dic[word]
            if fine_id != coarse_id:
                print(word)

def main(coarse_nn_config, fine_nn_config, coarse_data_config, fine_data_config):
    ana = Analysis(coarse_nn_config, fine_nn_config, coarse_data_config, fine_data_config)
    ana.check_table()

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0)
    args = parser.parse_args()
    seed = {'lstm_cell_size': 300,
            'word_dim': 300,
            'attribute_mat_size': 5
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

    if getpass.getuser() == "lujunyu":

        coarse_nn_config[
            'sr_path'] = '/home/lujunyu/repository/sentiment_coarse_model/sentiment/coarse_nn/coarse_atr_classifier_1pNw_bilstm/ckpt_bi_5mention_6.19/'
        coarse_data_config['train_source_file_path'] = '/home/lujunyu/dataset/yelp/yelp_lda_trainset.pkl'
        coarse_data_config['test_source_file_path'] = '/home/lujunyu/dataset/yelp/yelp_lda_testset.pkl'
        coarse_data_config[
            'train_data_file_path'] = '/home/lujunyu/repository/sentiment_coarse_model/sentiment/transfer_nn/coarse_train_data.pkl'
        coarse_data_config[
            'test_data_file_path'] = '/home/lujunyu/repository/sentiment_coarse_model/sentiment/transfer_nn/coarse_test_data.pkl'
        coarse_data_config[
            'wordembedding_file_path'] = '~/dataset/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin'
        coarse_data_config['stopwords_file_path'] = '~/dataset/semeval2016/stopwords.txt'
        coarse_data_config[
            'fine_sentences_file'] = '/home/lujunyu/repository/sentiment_coarse_model/sentiment/transfer_nn/fine_sentences_data.pkl'
        coarse_data_config[
            'dictionary'] = '/home/lujunyu/repository/sentiment_coarse_model/sentiment/util/word_dic/data_dictionary.pkl'

        fine_nn_config['sr_path'] = ''
        fine_data_config['train_source_file_path'] = '/home/lujunyu/dataset/semeval2016/absa_resturant_train.pkl'
        fine_data_config['test_source_file_path'] = '/home/lujunyu/dataset/semeval2016/absa_resturant_test.pkl'
        fine_data_config[
            'train_data_file_path'] = '/home/lujunyu/repository/sentiment_coarse_model/sentiment/transfer_nn/fine_train_data.pkl'
        fine_data_config[
            'test_data_file_path'] = '/home/lujunyu/repository/sentiment_coarse_model/sentiment/transfer_nn/fine_test_data.pkl'
        fine_data_config[
            'wordembedding_file_path'] = '~/dataset/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin'
        fine_data_config['stopwords_file_path'] = '~/dataset/semeval2016/stopwords.txt'
        fine_data_config[
            'dictionary'] = '/home/lujunyu/repository/sentiment_coarse_model/sentiment/util/word_dic/data_dictionary.pkl'

        fine_nn_config['reg_rage'] = reg_rate[args.num]
        fine_nn_config['lr'] = lr[args.num]
        # path of tensorboard files
        fine_nn_config['tfb_filePath'] = ''
        # path of meta data for fine attribute label
        fine_nn_config['tfb_fine_atr_metaPath'] = ''
        # path of meta data for fine coarse label
        fine_nn_config['tfb_coarse_atr_metaPath'] = ''

    elif getpass.getuser() == "liu121":
        coarse_nn_config[
            'sr_path'] = '/datastore/liu121/sentidata2/expdata/transfer/coarse_grain/model/ckpt_bi_5mention_6.19/'
        coarse_data_config['train_source_file_path'] = '/datastore/liu121/sentidata2/expdata/yelp/yelp_lda_trainset.pkl'
        coarse_data_config['test_source_file_path'] = '/datastore/liu121/sentidata2/expdata/yelp/yelp_lda_testset.pkl'
        coarse_data_config[
            'train_data_file_path'] = '/datastore/liu121/sentidata2/expdata/transfer/coarse_grain/data/coarse_train_data.pkl'
        coarse_data_config[
            'test_data_file_path'] = '/datastore/liu121/sentidata2/expdata/transfer/coarse_grain/data/coarse_test_data.pkl'
        coarse_data_config[
            'wordembedding_file_path'] = '/datastore/liu121/wordEmb/googlenews/GoogleNews-vectors-negative300.bin'
        coarse_data_config['stopwords_file_path'] = '/datastore/liu121/sentidata2/expdata/stopwords.txt'
        coarse_data_config[
            'fine_sentences_file'] = '/datastore/liu121/sentidata2/expdata/transfer/fine_grain/data/fine_sentences_data.pkl'
        coarse_data_config['dictionary'] = '/datastore/liu121/sentidata2/expdata/data_dictionary.pkl'

        fine_nn_config['sr_path'] = '/datastore/liu121/sentidata2/resultdata/transfer/model'
        fine_data_config[
            'train_source_file_path'] = '/datastore/liu121/sentidata2/expdata/semeval2016/absa_resturant_train.pkl'
        fine_data_config[
            'test_source_file_path'] = '/datastore/liu121/sentidata2/expdata/semeval2016/absa_resturant_test.pkl'
        fine_data_config[
            'train_data_file_path'] = '/datastore/liu121/sentidata2/expdata/transfer/fine_grain/data/fine_train_data.pkl'
        fine_data_config[
            'test_data_file_path'] = '/datastore/liu121/sentidata2/expdata/transfer/fine_grain/data/fine_test_data.pkl'
        fine_data_config[
            'wordembedding_file_path'] = '/datastore/liu121/wordEmb/googlenews/GoogleNews-vectors-negative300.bin'
        fine_data_config['stopwords_file_path'] = '/datastore/liu121/sentidata2/expdata/stopwords.txt'
        fine_data_config['dictionary'] = '/datastore/liu121/sentidata2/expdata/data_dictionary.pkl'

        fine_nn_config['reg_rage'] = reg_rate[args.num]
        fine_nn_config['lr'] = lr[args.num]
        fine_nn_config['tfb_filePath'] = '/datastore/liu121/sentidata2/resultdata/transfer/tfb/mat%s_reg%s_lr%s' \
                                         % (str(fine_nn_config['attribute_mat_size']), str(reg_rate[args.num]),
                                            str(lr[args.num]))
        fine_nn_config[
            'tfb_fine_atr_metaPath'] = '/datastore/liu121/sentidata2/expdata/meta/semeval2016_absa_resturant_label.tsv'
        fine_nn_config['tfb_coarse_atr_metaPath'] = '/datastore/liu121/sentidata2/expdata/meta/yelp_label.tsv'
        fine_nn_config['coarse_attributes_num'] = coarse_nn_config['attributes_num']

    main(coarse_nn_config, fine_nn_config, coarse_data_config, fine_data_config)