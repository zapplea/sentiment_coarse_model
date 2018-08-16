import getpass
import sys
if getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
elif getpass.getuser() == "lizhou":
    sys.path.append('/media/data2tb4/yibing2/sentiment_coarse_model/')

import os
import pickle
import sklearn
import operator
from pathlib import Path
import json
import argparse
import numpy as np

from sentiment.transfer_nn.ilp_1pNw.classifier import Classifier
from sentiment.util.coarse.atr_data_feeder import DataFeeder as coarse_DataFeeder
from sentiment.util.fine.atr_data_feeder import DataFeeder as fine_DataFeeder

# TODO: analyze init_data
# TODO: 1. mapping between labels
# TODO: 2. nearest neighbour of attribute mention vector
# TODO: check whether coarse and fine use the same table
class Analysis:
    def __init__(self, coarse_nn_config, fine_nn_config, coarse_data_config, fine_data_config, config_ana):
        # coarse_df = coarse_DataFeeder(coarse_data_config,coarse_nn_config)
        # fine_df = fine_DataFeeder(fine_data_config)
        # self.cl = Classifier(coarse_nn_config, fine_nn_config, coarse_df, fine_df)
        # self.coarse_nn_config = coarse_nn_config
        self.fine_nn_config = fine_nn_config

        self.coarse_data_config=coarse_data_config
        self.fine_data_config = fine_data_config

        self.config_ana = config_ana

        # with open(self.coarse_data_config['train_data_file_path'],'rb') as f:
        #     self.aspect_dic = pickle.load(f)
        #     self.coarse_word_dic=pickle.load(f)
        #     self.aspect_labels = pickle.load(f)
        #     self.coarse_sentences = pickle.load(f)
        #     self.coarse_table = pickle.load(f)
        #
        # with open(self.fine_data_config['train_data_file_path'],'rb') as f:
        #     self.attribute_dic = pickle.load(f)
        #     self.fine_word_dic = pickle.load(f)
        #     self.attribute_labels=pickle.load(f)
        #     self.fine_sentences = pickle.load(f)
        #     self.fine_table = pickle.load(f)
        # self.id2word={}
        # for word in self.fine_word_dic:
        #     self.id2word[self.fine_word_dic[word]]=word

    def check_table(self):
        print(len(self.coarse_word_dic))
        print(len(self.fine_word_dic))
        print('UNK id: ',self.coarse_word_dic['#UNK#'])
        print('PAD id: ',self.coarse_word_dic['#PAD#'])

    def transfer_data_generator(self):
        coarse_cl = self.cl.coarse_classifier()
        init_data = self.cl.transfer(coarse_cl)
        coarse_A = init_data['coarse_A']
        print(coarse_A[0])
        exit()
        return init_data

    def aspect_mention_vector_nearest_word(self,init_data):
        aspect_A = init_data['coarse_A']
        table = init_data['init_table']
        id2aspect={}
        for key in self.aspect_dic:
            id2aspect[self.aspect_dic[key]]=key
        map = []
        for i in range(len(id2aspect)-1):
            aspect_matrix = aspect_A[i]
            distance = sklearn.metrics.pairwise.pairwise_distances(aspect_matrix,table)
            map.append([])
            for j in range(distance.shape[0]):
                map[i].append([])
                for l in range(distance.shape[1]):
                    value = distance[j][l]
                    map[i][j].append((l,value))
                map[i][j] = sorted(map[i][j],key=operator.itemgetter(1))
        k_nearest={}
        for i in range(len(id2aspect)-1):
            label = id2aspect[i]
            k_nearest[label]={}
            for j in range(len(map[i])):
                k_nearest[label]['mention_%s' % str(j)]=[]
                for word in map[i][j][:self.config_ana['top_k']]:
                    k_nearest[label]['mention_%s'%str(j)].append(self.id2word[word[0]])
        report_filePath = os.path.join(self.config_ana['report'],'aspect_nearest_top%s.json'%str(self.config_ana['top_k']))
        with open(report_filePath,'w+') as f:
            json.dump(k_nearest, f, indent=4, sort_keys=False)


    def attribute_mention_vector_nearest_word(self,init_data):
        attribute_A = init_data['init_A']
        table = init_data['init_table']
        map = []
        id2attribute = {}
        for key in self.attribute_dic:
            id2attribute[self.attribute_dic[key]] = key
        for i in range(len(id2attribute)-1):
            attribute_matrix = attribute_A[i]
            distance = sklearn.metrics.pairwise.pairwise_distances(attribute_matrix, table)
            map.append([])
            for j in range(distance.shape[0]):
                map[i].append([])
                for l in range(distance.shape[1]):
                    value = distance[j][l]
                    map[i][j].append((l, value))
                map[i][j] = sorted(map[i][j], key=operator.itemgetter(1))
        k_nearest = {}
        for i in range(len(id2attribute)-1):
            label = id2attribute[i]
            k_nearest[label] = {}
            for j in range(len(map[i])):
                k_nearest[label]['mention_%s' % str(j)] = []
                for word in map[i][j][:self.config_ana['top_k']]:
                    k_nearest[label]['mention_%s' % str(j)].append(self.id2word[word[0]])
        report_filePath = os.path.join(self.config_ana['report'],
                                       'attribute_nearest_top%s.json' % str(self.config_ana['top_k']))
        with open(report_filePath, 'w+') as f:
            json.dump(k_nearest, f, indent=4, sort_keys=False)

    def check_fine_dataset(self):
        with open(self.fine_data_config['train_data_file_path'], 'rb') as f:
            attribute_dic,word_dic,label,sentence,word_embed = pickle.load(f)
            print('max fine train data id: ',np.amax(sentence))
            print(np.max(label))
        with open(self.fine_data_config['test_data_file_path'], 'rb') as f:
            label,sentence = pickle.load(f)
            print('max fine test data id: ',np.amax(sentence))
            print(np.max(label))

def main(coarse_nn_config, fine_nn_config, coarse_data_config, fine_data_config):
    if getpass.getuser()=="liu121":
        config_ana={'top_k':5,
                    'report':'/datastore/liu121/sentidata2/resultdata/analysis/'}
    elif getpass.getuser() == "lizhou":
        config_ana = {'top_k': 5,
                      'report': '/media/data2tb4/yibing2/datastore/sentidata2/resultdata/analysis/'}
    path = Path(config_ana['report'])
    if not path.exists():
        path.mkdir(parents=True,exist_ok=True)


    ana = Analysis(coarse_nn_config, fine_nn_config, coarse_data_config, fine_data_config, config_ana)
    ana.check_fine_dataset()
    # init_data = ana.transfer_data_generator()
    # ana.aspect_mention_vector_nearest_word(init_data)
    # ana.attribute_mention_vector_nearest_word(init_data)

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hype_num', type=int, default=0,help='the num of lr and reg_rate')
    parser.add_argument('--dataset',type=str,default='0', help='which dataset will be used')
    parser.add_argument('--train_mod',type=str,default='v2', help='the version of loss function')
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
        'lookup_table_words_num': 34934,  # 2074276 for Chinese word embedding
        'padding_word_index': 34933,  # the index of #PAD# in word embeddings list
        # 'unk_word_index': 41484,
        # flexible parameter
        'reg_rate': 0.003,
        'lr': 0.0003,  # learing rate
        'atr_pred_threshold': 0,
        # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
        'attribute_loss_theta': 3.0,
        'aspect_prob_threshold': 0.2,
        'keep_prob_lstm': 0.5,
        'complement': 0,
        'train_mod':args.train_mod
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
        'lookup_table_words_num': 34934,
        'padding_word_index': coarse_nn_config['padding_word_index'],  # the index of #PAD# in word embeddings list
        # 'unk_word_index': coarse_nn_config['unk_word_index'],
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

    reg_rate = [3E-4, 3E-3, 3E-2, 3E-1, 3E-4, 3E-4, 3E-4]
    lr = [3E-3, 3E-3, 3E-3, 3E-3, 3E-4, 3E-5, 3E-6]

    if getpass.getuser() == "liu121":
        coarse_nn_config['sr_path'] = '/datastore/liu121/sentidata2/resultdata/coarse_nn/model/ckpt%s_dataset%s_reg%s_lr%s_aspect%s_mat%s/'\
                                      %(str(args.train_mod),str(args.dataset),str(reg_rate[args.hype_num]),str(lr[args.hype_num]),str(coarse_nn_config['attributes_num']),str(seed['attribute_mat_size']))

        coarse_data_config['train_source_file_path'] = '/datastore/liu121/sentidata2/expdata/yelp/yelp_lda_trainset.pkl'
        coarse_data_config['test_source_file_path'] = '/datastore/liu121/sentidata2/expdata/yelp/yelp_lda_testset.pkl'
        coarse_data_config[
            'train_data_file_path'] = '/datastore/liu121/sentidata2/expdata/coarse_data/coarse_train_data_v%s.pkl'%str(args.dataset)
        coarse_data_config[
            'test_data_file_path'] = '/datastore/liu121/sentidata2/expdata/coarse_data/coarse_test_data_v%s.pkl'%str(args.dataset)
        coarse_data_config[
            'wordembedding_file_path'] = '/datastore/liu121/wordEmb/googlenews/GoogleNews-vectors-negative300.bin'
        # coarse_data_config['stopwords_file_path'] = '/datastore/liu121/sentidata2/expdata/stopwords.txt'
        coarse_data_config[
            'fine_sentences_file'] = '/datastore/liu121/sentidata2/expdata/fine_data/fine_sentences_data.pkl'
        # coarse_data_config['dictionary'] = '/datastore/liu121/sentidata2/expdata/data_dictionary.pkl'

        fine_nn_config['sr_path'] = '/datastore/liu121/sentidata2/resultdata/coarse_nn/model/ckpt%s_dataset%s_reg%s_lr%s_aspect%s_mat%s/'\
                                    %(str(args.train_mod),str(args.dataset),str(reg_rate[args.hype_num]),str(lr[args.hype_num]),str(fine_nn_config['attributes_num']),str(seed['attribute_mat_size']))
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

        fine_nn_config['reg_rage'] = reg_rate[args.hype_num]
        fine_nn_config['lr'] = lr[args.hype_num]
        fine_nn_config['tfb_filePath'] = fine_nn_config['sr_path']
        fine_nn_config['coarse_attributes_num'] = coarse_nn_config['attributes_num']

    main(coarse_nn_config, fine_nn_config, coarse_data_config, fine_data_config)

    # elif getpass.getuser() == "lizhou":
    #     print('lizhou')
    #     coarse_nn_config[
    #         'sr_path'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/transfer/coarse_grain/model/ckpt_bi_5mention_6.19/'
    #     coarse_data_config['train_source_file_path'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/yelp/yelp_lda_trainset.pkl'
    #     coarse_data_config['test_source_file_path'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/yelp/yelp_lda_testset.pkl'
    #     coarse_data_config[
    #         'train_data_file_path'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/transfer/coarse_grain/data/coarse_train_data.pkl'
    #     coarse_data_config[
    #         'test_data_file_path'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/transfer/coarse_grain/data/coarse_test_data.pkl'
    #     coarse_data_config[
    #         'wordembedding_file_path'] = '/media/data2tb4/yibing2/datastore/wordEmb/googlenews/GoogleNews-vectors-negative300.bin'
    #     coarse_data_config['stopwords_file_path'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/stopwords.txt'
    #     coarse_data_config[
    #         'fine_sentences_file'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/transfer/fine_grain/data/fine_sentences_data.pkl'
    #     coarse_data_config['dictionary'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/data_dictionary.pkl'
    #
    #     fine_nn_config['sr_path'] = '/media/data2tb4/yibing2/datastore/sentidata2/resultdata/transfer/model'
    #     fine_data_config[
    #         'train_source_file_path'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/semeval2016/absa_resturant_train.pkl'
    #     fine_data_config[
    #         'test_source_file_path'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/semeval2016/absa_resturant_test.pkl'
    #     fine_data_config[
    #         'train_data_file_path'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/transfer/fine_grain/data/fine_train_data.pkl'
    #     fine_data_config[
    #         'test_data_file_path'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/transfer/fine_grain/data/fine_test_data.pkl'
    #     fine_data_config[
    #         'wordembedding_file_path'] = '/media/data2tb4/yibing2/datastore/wordEmb/googlenews/GoogleNews-vectors-negative300.bin'
    #     fine_data_config['stopwords_file_path'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/stopwords.txt'
    #     fine_data_config['dictionary'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/data_dictionary.pkl'
    #
    #     fine_nn_config['reg_rage'] = reg_rate[args.num]
    #     fine_nn_config['lr'] = lr[args.num]
    #     fine_nn_config['tfb_filePath'] = '/media/data2tb4/yibing2/datastore/sentidata2/resultdata/transfer/tfb/mat%s_reg%s_lr%s' \
    #                                      % (str(fine_nn_config['attribute_mat_size']), str(reg_rate[args.num]),
    #                                         str(lr[args.num]))
    #     fine_nn_config[
    #         'tfb_fine_atr_metaPath'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/meta/semeval2016_absa_resturant_label.tsv'
    #     fine_nn_config['tfb_coarse_atr_metaPath'] = '/media/data2tb4/yibing2/datastore/sentidata2/expdata/meta/yelp_label.tsv'
    #     fine_nn_config['coarse_attributes_num'] = coarse_nn_config['attributes_num']

