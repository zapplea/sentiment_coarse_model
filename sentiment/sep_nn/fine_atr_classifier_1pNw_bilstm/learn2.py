import os
import sys
import getpass
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif getpass.getuser() == 'lujunyu':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"  ## 0
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import argparse
from pathlib import Path
from sentiment.sep_nn.fine_atr_classifier_1pNw_bilstm.classifier2 import Classifier
from sentiment.util.fine.atr_data_feeder import DataFeeder

def main(config):
    dg = DataFeeder(config)
    cl = Classifier(config, dg)
    cl.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.add_argument('--hype_num',str=int,default=0)
    reg_rate=[1E-5]
    lr=[1E-4]
    config = {
        'fine_train_data_file': '/datastore/liu121/sentidata2/expdata/fine_data/fine_train_data.pkl',
        'fine_test_data_file': '/datastore/liu121/sentidata2/expdata/fine_data/fine_test_data.pkl',
        'wordembedding_file_path': '/datastore/liu121/wordEmb/googlenews/GoogleNews-vectors-negative300.bin',

        'lstm_cell_size': 300,
        'word_dim': 300,
        'attribute_dim': 300,
        'lookup_table_words_num': 34934,  # 34934,2074276 for Chinese word embedding
        'padding_word_index': 34933,  # 34933,the index of #PAD# in word embeddings list
        'epoch': 1000,
        'words_num': 40,
        'atr_pred_threshold': 0,
        'is_mat': True,

        # fixed parameter
        'attribute_mat_size': 5,  # number of attribute mention prototypes in a attribute matrix
        'attributes_num': 12,
        'batch_size': 200,
        # flexible parameter
        'reg_rate': reg_rate[args.hype_num],
        'attribute_loss_theta': 3.0,
        'lr': lr[args.hype_num],  # learing rate
        'keep_prob_lstm': 0.5,
        'top_k_data': -1,
        'early_stop_limit':100
    }
    config['report_filePath']='/datatstore/liu121/sentidata2/resultdata/fine_nn/report/report_reg%s_lr%s_mat%s.info'\
                              %(str(reg_rate[args.hype_num]),str(lr[args.hype_num]),str(config['attribute_mat_size']))
    config['tfb_filePath'] = '/datatstore/liu121/sentidata2/resultdata/fine_nn/model/ckpt_reg%s_lr%s_mat%s'\
                              %(str(reg_rate[args.hype_num]),str(lr[args.hype_num]),str(config['attribute_mat_size']))
    path=Path(config['tfb_filePath'])
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    main(config)