import os
import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"  ## 0
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
elif getpass.getuser() == 'lizhou':
    sys.path.append('/media/data2tb4/yibing2/sentiment_coarse_model/')
import argparse

from sentiment.coarse_nn.coarse_atr_classifier_1pNw_bilstm.classifier2 import Classifier
from sentiment.util.coarse.atr_data_generator2 import DataGenerator
from pathlib import Path

def main(nn_config,data_config):
    # nn_config.update(data_config)
    dg = DataGenerator(data_config,nn_config)
    cl = Classifier(nn_config, dg)
    cl.train()

# TODO: check whether the program is correct, eg. how to process padded words.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--dataset',type=str, default='1.5')
    args = parser.parse_args()
    seed = {'lstm_cell_size': 300,
            'word_dim': 300,
            'attribute_mat_size': 5,
            'attributes_num': 7,
            }
    reg_rate = [3E-4, 3E-3, 3E-2, 3E-1, 3E-4, 3E-4, 3E-4]
    lr =       [3E-3, 3E-3, 3E-3, 3E-3, 3E-4, 3E-5, 3E-6]
    nn_config = {
        # fixed parameter
        'attributes_num': seed['attributes_num'],
        'attribute_dim': seed['word_dim'],
        'attribute_mat_size': seed['attribute_mat_size'],
        # number of attribute mention prototypes in a attribute matrix
        'max_review_length': 30,# TODO: check max review length
        'words_num': 40,
        'word_dim': seed['word_dim'],
        'is_mat': True,
        'epoch': 1000,
        'batch_size': 20,
        'lstm_cell_size': seed['lstm_cell_size'],
        'lookup_table_words_num': 34934,  # 2074276 for Chinese word embedding
        'padding_word_index': 34933,  # the index of #PAD# in word embeddings list
        'unk_word_index': 34932, # TODO: check wheter the unk have been recognized.
        # flexible parameter
        'reg_rate': reg_rate[args.num],
        'lr': lr[args.num],  # learing rate
        'atr_pred_threshold': 0,
        # if score of attribute is larger than atr_pred_threshold, the attribute exists in the sentence
        #'attribute_loss_theta': 2.0,
        'aspect_prob_threshold': 0.16,
        'keep_prob_lstm': 0.5,
        'complement': 0,
        'model_save_path': '/datastore/liu121/sentidata2/resultdata/coarse_nn/model/ckpt_dataset%s_reg%s_lr%s_aspect%s_mat%s/'%(args.dataset,str(reg_rate[args.num]),str(lr[args.num]),str(seed['attributes_num']),str(seed['attribute_mat_size'])),
        'tfb_filePath':'/datastore/liu121/sentidata2/resultdata/coarse_nn/model/ckpt_dataset%s_reg%s_lr%s_aspect%s_mat%s/'%(args.dataset,str(reg_rate[args.num]),str(lr[args.num]),str(seed['attributes_num']),str(seed['attribute_mat_size'])),
        # 'sr_path': '',
        'train_mod':'sigmoid',
        'early_stop_limit':20
    }
    path = Path(nn_config['model_save_path'])
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    nn_config['model_save_path']=nn_config['model_save_path']+'model.ckpt'

    data_config = {
        'train_source_file_path': '/datastore/liu121/sentidata2/expdata/yelp/yelp_lda_trainset.pkl',
        'train_data_file_path': '/datastore/liu121/sentidata2/expdata/coarse_data/coarse_train_data_v%s.pkl'%args.dataset,
        'test_source_file_path': '/datastore/liu121/sentidata2/expdata/yelp/yelp_lda_testset.pkl',
        'test_data_file_path': '/datastore/liu121/sentidata2/expdata/coarse_data/coarse_test_data_v%s.pkl'%args.dataset,
        'wordembedding_file_path': '/datastore/liu121/wordEmb/googlenews/GoogleNews-vectors-negative300.bin',
        'stopwords_file_path': '/datastore/liu121/sentidata2/expdata/stopwords.txt',
        'attributes_num': nn_config['attributes_num'],
        'batch_size': nn_config['batch_size'],
        'words_num': nn_config['words_num'],
        'padding_word_index': nn_config['padding_word_index'],
        'word_dim': seed['word_dim'],
        'dictionary':'/datastore/liu121/sentidata2/expdata/data_dictionary.pkl'
    }
    # for nn_config in nn_configs:
    #     print(nn_config)
    #     main(nn_config,data_config)
    main(nn_config, data_config)