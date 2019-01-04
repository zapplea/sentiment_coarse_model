import getpass
import sys
if getpass.getuser() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model/')
elif getpass.getuser() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model/')
elif getpass.getuser() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model/')
import tensorflow as tf
import argparse
import pickle
import numpy as np

from aic.elmo.elmo_train import load_vocab
from aic.elmo.elmo_net import LanguageModel

def build(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)
    # define the options
    batch_size = 5  # batch size for each GPU
    n_gpus = 2

    # number of tokens in training data (this for 1B Word Benchmark)
    # n_train_tokens = 768648884
    n_train_tokens = 20539032

    options = {
        'bidirectional': True,

        'share_embedding_softmax': True,

        'dropout': 0.1,

        'lstm': {
            'cell_clip': 3,
            'dim': 4096,
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 300,
            'use_skip_connections': True},

        'all_clip_norm_val': 10.0,

        'n_epochs': 10,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 210,
        'n_negative_samples_batch': 4800,
        'initial_file_dir':'/datastore/liu121/sentidata2/result/bilm/',
    }

    config = tf.ConfigProto(allow_soft_placement=True)
    graph = tf.Graph()
    options['graph'] = graph
    with graph.as_default():
        with tf.device('/gpu:0'), tf.variable_scope('elmo'):
            LanguageModel(options, False)
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()

    with tf.Session(graph =graph ,config=config) as sess:
        model_file = tf.train.latest_checkpoint(options['initial_file_dir'])
        loader.restore(sess, model_file)
        extract(graph,sess)

def extract(model,sess):
    vars_dic = {}
    vars = model.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for var in vars:
        print(var.name)
        value = sess.run(var)
        if var.name.find('embedding:0'):
            shape = value.shape
            stdv = 1 / np.sqrt(shape[-1]).astype(str(value.dtype))
            init = np.random.uniform(size = (shape[-1],), low=-stdv, high=stdv).astype(str(value.dtype))
            value = np.delete(value,(0,1,2),axis=0)
            np.insert(value,0,init,0)
            print(var.name,' : ',value.shape)
        vars_dic[var.name] = value
    with open('/datastore/liu121/sentidata2/data/aic2018/elmo_weights/elmo_weights.pkl','wb') as f:
        pickle.dump(vars_dic,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', help='Vocabulary file',default='/datastore/liu121/sentidata2/data/bilm/vocab_aic.txt')
    args = parser.parse_args()
    build(args)