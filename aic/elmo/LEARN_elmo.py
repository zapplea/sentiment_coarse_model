import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model/')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model/')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model/')
import argparse
import numpy as np

from aic.elmo.elmo_train import train, load_options_latest_checkpoint, load_vocab
from aic.elmo.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)
    table = vocab._id_to_word
    # define the options
    batch_size = 128  # batch size for each GPU
    n_gpus = args.gpu_num

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 768648884
    # TODO: there is not epoch
    options = {
        'bidirectional': True,

        'dropout': 0.1,

        'lstm': {
            'cell_clip': 3,
            'dim': 4096,
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 512,
            'use_skip_connections': True},

        'all_clip_norm_val': 10.0,

        'n_epochs': 10,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 20,
        'n_negative_samples_batch': 8192,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                  shuffle_on_load=True)
    # TODO: analyze data
    data_gen = data.iter_batches(128 * 3, 20)
    for batch_no, batch in enumerate(data_gen, start=1):
        # slice the input in the batch for the feed_dict
        X = batch
        # TODO: analyze 1. how to generate next id (+1)
        # TODO:         2. how to generate reverse.
        # TODO:         3. how to process pad (?cut the sentence based on max sentence length?)
        print(X.keys())
        print('=============')
        print(X['token_ids'])
        print(X['next_token_ids'])
        print('=============')
        print(X['token_ids_reverse'])
        exit()

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--gpu_num',type=int,default=3)

    args = parser.parse_args()
    main(args)

