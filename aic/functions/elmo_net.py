'''
Train and test bidirectional language models.
'''
import tensorflow as tf
import numpy as np

from tensorflow.python.ops.init_ops import glorot_uniform_initializer

from aic.elmo.data import InvalidNumberOfCharacters

DTYPE = 'float32'
DTYPE_INT = 'int64'

tf.logging.set_verbosity(tf.logging.INFO)

class LanguageModel(object):
    '''
    A class to build the tensorflow computational graph for NLMs

    All hyperparameters and model configuration is specified in a dictionary
    of 'options'.

    is_training is a boolean used to control behavior of dropout layers
        and softmax.  Set to False for testing.

    The LSTM cell is controlled by the 'lstm' key in options
    Here is an example:

     'lstm': {
      'cell_clip': 5,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 5,
      'projection_dim': 512,
      'use_skip_connections': True},

        'projection_dim' is assumed token embedding size and LSTM output size.
        'dim' is the hidden state size.
        Set 'dim' == 'projection_dim' to skip a projection layer.
    '''

    def __init__(self, options, is_training):
        self.options = {
            'bidirectional': True,

            'share_embedding_softmax':True,

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
            'unroll_steps': 210,
            'n_negative_samples_batch': 4800,
        }
        self.options.update(options)
        self.is_training = is_training

    def mask_for_token_ids(self,token_ids):
        """
        :param token_ids: shape = (batch size, max sentence length) 
        :return: (batch size, max sentence length)
        """
        paddings = tf.ones_like(token_ids, dtype='int32') * self.options['padding_word_index']
        condition = tf.equal(paddings, token_ids)
        mask = tf.where(condition, tf.zeros_like(token_ids, dtype='float32'), tf.ones_like(token_ids, dtype='float32'))
        return mask

    def bilstm(self,X,seq_len):
        """
        
        :param X: shape = (batch size, max sentence len, word dim) 
        :param seq_len: 
        :return: 
        """
        lstm_dim = self.options['lstm']['dim']
        projection_dim = self.options['lstm']['projection_dim']
        n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        dropout = self.options['dropout']
        keep_prob = 1.0 - dropout

        cell_clip = self.options['lstm'].get('cell_clip')
        proj_clip = self.options['lstm'].get('proj_clip')
        use_skip_connections = self.options['lstm'].get(
            'use_skip_connections')

        # use MultiRNNCells to implement multi-layers biLSTM
        # lm_embeddings.shape = (n_lstm_layers+1,batch size, words num, 2*lstm_dim)
        lm_embeddings=[]
        lm_embeddings.append(tf.concat([X,X],axis=2))
        outputs_fw=X
        outputs_bw=tf.reverse_sequence(X,seq_len,seq_axis=1,batch_axis=0)
        for i in range(n_lstm_layers):
            lstm_cell_ls = []
            for name in ['fw','bw']:
                if projection_dim < lstm_dim:
                    # are projecting down output
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim, num_proj=projection_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip,name='layer%d'%i)
                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip)

                if use_skip_connections:
                    # ResidualWrapper adds inputs to outputs
                    if i == 0:
                        # don't add skip connection from token embedding to
                        # 1st layer output
                        pass
                    else:
                        # add a skip connection
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

                # add dropout
                if self.is_training:
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                              input_keep_prob=keep_prob)
                lstm_cell_ls.append(lstm_cell)
            # TODO: check how to control name of kernel
            # outputs_fw.shape = (batch size, words num, lstm_dim)
            outputs_fw, _ = tf.nn.dynamic_rnn(lstm_cell_ls[0],inputs=outputs_fw,sequence_length=seq_len,dtype='float32',scope='bilstm/fw')
            outputs_bw, _ = tf.nn.dynamic_rnn(lstm_cell_ls[1],inputs=outputs_bw,sequence_length=seq_len,dtype='float32',scope='bilstm/bw')
            # shape = (n_lstm_layers+1,batch size, words num, 2*lstm_dim)
            lm_embeddings.append(tf.concat([outputs_fw, outputs_bw], axis=2))
            # lm_embeddings.shape=(batch size, lstm layers+1, max sentence length, 2*lstm dim)
            lm_embeddings = tf.transpose(lm_embeddings,perm=[1,0,2,3])

        return lm_embeddings

    def weight_layers(self, name, bilm_ops,reg, l2_coef=None,
                      use_top_only=False, do_layer_norm=False):
        '''
        Weight the layers of a biLM with trainable scalar weights to
        compute ELMo representations.

        For each output layer, this returns two ops.  The first computes
            a layer specific weighted average of the biLM layers, and
            the second the l2 regularizer loss term.
        The regularization terms are also add to tf.GraphKeys.REGULARIZATION_LOSSES 

        Input:
            name = a string prefix used for the trainable variable names
            bilm_ops = the tensorflow ops returned to compute internal
                representations from a biLM.  This is the return value
                from BidirectionalLanguageModel(...)(ids_placeholder)
            l2_coef: the l2 regularization coefficient $\lambda$.
                Pass None or 0.0 for no regularization.
            use_top_only: if True, then only use the top layer.
            do_layer_norm: if True, then apply layer normalization to each biLM
                layer before normalizing

        Output:
            {
                'weighted_op': op to compute weighted average for output,
                'regularization_op': op to compute regularization term
            }
        '''

        def _l2_regularizer(weights):
            if l2_coef is not None:
                return l2_coef * tf.reduce_sum(tf.square(weights))
            else:
                return 0.0

        # Get ops for computing LM embeddings and mask
        # lm_embeddings.shape=(batch size, lstm layers+1, max sentence length, 2*lstm dim)
        lm_embeddings = bilm_ops['lm_embeddings']
        # mask.shape = (batch, max sentence length)
        mask = bilm_ops['mask']

        n_lm_layers = int(lm_embeddings.get_shape()[1])
        lm_dim = int(lm_embeddings.get_shape()[3])

        with tf.control_dependencies([lm_embeddings, mask]):
            # Cast the mask and broadcast for layer use.
            mask_float = tf.cast(mask, 'float32')
            # shape = (batch, max sentence length, proj_dim)
            broadcast_mask = tf.tile(tf.expand_dims(mask_float, axis=-1),multiples=[1,1,self.options['lstm']['projection_dim']])

            def _do_ln(x):
                # do layer normalization excluding the mask
                # x.shape = (batch size, max sentence length, 2*lstm dim)
                x_masked = x * broadcast_mask
                N = tf.reduce_sum(mask_float) * lm_dim
                mean = tf.reduce_sum(x_masked) / N
                variance = tf.reduce_sum(((x_masked - mean) * broadcast_mask) ** 2
                                         ) / N
                return tf.nn.batch_normalization(
                    x, mean, variance, None, None, 1E-12
                )

            if use_top_only:
                # (lstm layers+1, batch size, 1, max sentence length, 2*lstm dim)
                layers = tf.split(lm_embeddings, n_lm_layers, axis=1)
                # just the top layer
                # shape = (batch size, max sentence length, 2*lstm dim)
                sum_pieces = tf.squeeze(layers[-1], squeeze_dims=1)
                # no regularization
                reg['elmo'].append(0.0)
            else:
                W = tf.get_variable(
                    '{}_ELMo_W'.format(name),
                    shape=(n_lm_layers,),
                    initializer=tf.zeros_initializer,
                    regularizer=_l2_regularizer,
                    trainable=True,
                )
                reg['elmo'].append(W)
                # normalize the weights
                normed_weights = tf.split(
                    tf.nn.softmax(W + 1.0 / n_lm_layers), n_lm_layers
                )
                # split LM layers
                # layers.shape = (lstm layers+1, batch size, 1, max sentence length, 2*lstm dim)
                layers = tf.split(lm_embeddings, n_lm_layers, axis=1)

                # compute the weighted, normalized LM activations
                pieces = []
                for w, t in zip(normed_weights, layers):
                    # t.shape = (batch size, 1, max sentence length, 2*lstm dim)
                    if do_layer_norm:
                        pieces.append(w * _do_ln(tf.squeeze(t, squeeze_dims=1)))
                    else:
                        pieces.append(w * tf.squeeze(t, squeeze_dims=1))
                sum_pieces = tf.add_n(pieces)

            # scale the weighted sum by gamma
            gamma = tf.get_variable(
                '{}_ELMo_gamma'.format(name),
                shape=(1,),
                initializer=tf.ones_initializer,
                regularizer=None,
                trainable=True,
            )
            # shape = (batch size, max sentence length, 2*lstm dim)
            weighted_lm_layers = sum_pieces * gamma

            return weighted_lm_layers




