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

from aic.elmo_test.elmo_train import train, load_options_latest_checkpoint, load_vocab
from aic.elmo_test.data import BidirectionalLMDataset
import pickle

def store(args,filepath):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)
    # define the options
    batch_size = 128  # batch size for each GPU
    n_gpus = 3
    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 768648884
    options = {
        'bidirectional': True,

        'char_cnn': {'activation': 'relu',
                     'embedding': {'dim': 16},
                     'filters': [[1, 32],
                                 [2, 32],
                                 [3, 64],
                                 [4, 128],
                                 [5, 256],
                                 [6, 512],
                                 [7, 1024]],
                     'max_characters_per_token': 50,
                     'n_characters': 261,
                     'n_highway': 2},

        'dropout': 0.1,

        'lstm': {
            'cell_clip': 3,
            'dim': 4096,
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 512,
            'use_skip_connections': True},

        'all_clip_norm_val': 10.0,

        'n_epochs': 1,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 20,
        'n_negative_samples_batch': 8192,
    }
    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                  shuffle_on_load=True)
    data_gen = data.iter_batches(batch_size * n_gpus, 20)

    batch_size = options['batch_size']
    unroll_steps = options['unroll_steps']
    n_train_tokens = options.get('n_train_tokens', 768648884)
    print('train tokens total:',n_train_tokens)
    n_tokens_per_batch = batch_size * unroll_steps * n_gpus
    print('tokens per batch: ',n_tokens_per_batch)
    n_batches_per_epoch = int(n_train_tokens / n_tokens_per_batch)
    print('batches per epoch: ',n_batches_per_epoch)
    n_batches_total = options['n_epochs'] * n_batches_per_epoch
    print('batches num total: ',n_batches_total)
    exit()

    batchs = []
    for batch_no, batch in enumerate(data_gen, start=1):
        batchs.append(batch)
        if batch_no == n_batches_total:
            # done training!
            break
    print('batchs.shape: ', len(batchs))
    with open(filepath,'wb') as f:
        pickle.dump(vocab,f)
        pickle.dump(batchs,f)

def analysis(filepath):
    with open(filepath,'rb') as f:
        vocab = pickle.load(f)
        batches = pickle.load(f)

    for batch in batches:
        print('#########################################')
        inputs = batch['token_ids']
        targets = batch['next_token_id']
        for i in range(5):
            sentence = []
            for scalar in inputs[i]:
                sentence.append(vocab._id_to_word[scalar])
            print('inputs[%d]: %d' % (i, len(sentence)), sentence)

            sentence = []
            for scalar in targets[i]:
                sentence.append(vocab._id_to_word[scalar])
            print('targets[%d]: %d' % (i, len(sentence)), sentence)
        print('==================')
        for i in range(-5, -1):
            sentence = []
            for scalar in inputs[i]:
                sentence.append(vocab._id_to_word[scalar])
            print('inputs[%d]: %d' % (i, len(sentence)), sentence)

            sentence = []
            for scalar in targets[i]:
                sentence.append(vocab._id_to_word[scalar])
            print('targets[%d]: %d' % (i, len(sentence)), sentence)
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')

    args = parser.parse_args()
    store(args,'/datastore/liu121/sentidata2/data/bilm/data.pkl')

