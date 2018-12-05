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
        self.graph = options['graph']
        self.options = options
        self.is_training = is_training
        self.bidirectional = options.get('bidirectional', False)

        # use word or char inputs?
        self.char_inputs = 'char_cnn' in self.options

        # for the loss function
        self.share_embedding_softmax = options.get(
            'share_embedding_softmax', False)
        if self.char_inputs and self.share_embedding_softmax:
            raise ValueError("Sharing softmax and embedding weights requires "
                             "word input")

        self.sample_softmax = options.get('sample_softmax', True)

        self._build()
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(var.name)
            print(var.get_shape())
        exit()

    def _build_word_embeddings(self):
        n_tokens_vocab = self.options['n_tokens_vocab']
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        # LSTM options
        projection_dim = self.options['lstm']['projection_dim']

        # the input token_ids and word embeddings
        self.token_ids = tf.placeholder(DTYPE_INT,
                                        shape=(batch_size, unroll_steps),
                                        name='token_ids')
        # the word embeddings
        with tf.device("/cpu:0"):
            # the word embeddings table
            self.embedding_weights = tf.get_variable(
                "embedding", [n_tokens_vocab, projection_dim],
                dtype=DTYPE,
            )
            # shape=(batch_size, unroll_steps)
            self.embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                    self.token_ids)

        # if a bidirectional LM then make placeholders for reverse
        # model and embeddings
        if self.bidirectional:
            self.token_ids_reverse = tf.placeholder(DTYPE_INT,
                                                    shape=(batch_size, unroll_steps),
                                                    name='token_ids_reverse')
            with tf.device("/cpu:0"):
                self.embedding_reverse = tf.nn.embedding_lookup(
                    self.embedding_weights, self.token_ids_reverse)

    def _build_word_char_embeddings(self):
        '''
        options contains key 'char_cnn': {

        'n_characters': 262,

        # includes the start / end characters
        'max_characters_per_token': 50,

        'filters': [
            [1, 32],
            [2, 32],
            [3, 64],
            [4, 128],
            [5, 256],
            [6, 512],
            [7, 512]
        ],
        'activation': 'tanh',

        # for the character embedding
        'embedding': {'dim': 16}

        # for highway layers
        # if omitted, then no highway layers
        'n_highway': 2,
        }
        '''
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']
        projection_dim = self.options['lstm']['projection_dim']

        cnn_options = self.options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        if n_chars != 261:
            raise InvalidNumberOfCharacters(
                "Set n_characters=261 for training see the README.md"
            )
        if cnn_options['activation'] == 'tanh':
            activation = tf.nn.tanh
        elif cnn_options['activation'] == 'relu':
            activation = tf.nn.relu

        # the input character ids
        self.tokens_characters = tf.placeholder(DTYPE_INT,
                                                shape=(batch_size, unroll_steps, max_chars),
                                                name='tokens_characters')
        # the character embeddings
        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                "char_embed", [n_chars, char_embed_dim],
                dtype=DTYPE,
                initializer=tf.random_uniform_initializer(-1.0, 1.0)
            )
            # shape (batch_size, unroll_steps, max_chars, embed_dim)
            self.char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                         self.tokens_characters)

            if self.bidirectional:
                self.tokens_characters_reverse = tf.placeholder(DTYPE_INT,
                                                                shape=(batch_size, unroll_steps, max_chars),
                                                                name='tokens_characters_reverse')
                self.char_embedding_reverse = tf.nn.embedding_lookup(
                    self.embedding_weights, self.tokens_characters_reverse)

        # the convolutions
        def make_convolutions(inp, reuse):
            with tf.variable_scope('CNN', reuse=reuse) as scope:
                convolutions = []
                for i, (width, num) in enumerate(filters):
                    if cnn_options['activation'] == 'relu':
                        # He initialization for ReLU activation
                        # with char embeddings init between -1 and 1
                        # w_init = tf.random_normal_initializer(
                        #    mean=0.0,
                        #    stddev=np.sqrt(2.0 / (width * char_embed_dim))
                        # )

                        # Kim et al 2015, +/- 0.05
                        w_init = tf.random_uniform_initializer(
                            minval=-0.05, maxval=0.05)
                    elif cnn_options['activation'] == 'tanh':
                        # glorot init
                        w_init = tf.random_normal_initializer(
                            mean=0.0,
                            stddev=np.sqrt(1.0 / (width * char_embed_dim))
                        )
                    w = tf.get_variable(
                        "W_cnn_%s" % i,
                        [1, width, char_embed_dim, num],
                        initializer=w_init,
                        dtype=DTYPE)
                    b = tf.get_variable(
                        "b_cnn_%s" % i, [num], dtype=DTYPE,
                        initializer=tf.constant_initializer(0.0))

                    conv = tf.nn.conv2d(
                        inp, w,
                        strides=[1, 1, 1, 1],
                        padding="VALID") + b
                    # now max pool
                    conv = tf.nn.max_pool(
                        conv, [1, 1, max_chars - width + 1, 1],
                        [1, 1, 1, 1], 'VALID')

                    # activation
                    conv = activation(conv)
                    conv = tf.squeeze(conv, squeeze_dims=[2])

                    convolutions.append(conv)

            return tf.concat(convolutions, 2)

        # for first model, this is False, for others it's True
        reuse = tf.get_variable_scope().reuse
        embedding = make_convolutions(self.char_embedding, reuse)

        self.token_embedding_layers = [embedding]

        if self.bidirectional:
            # re-use the CNN weights from forward pass
            embedding_reverse = make_convolutions(
                self.char_embedding_reverse, True)

        # for highway and projection layers:
        #   reshape from (batch_size, n_tokens, dim) to
        n_highway = cnn_options.get('n_highway')
        use_highway = n_highway is not None and n_highway > 0
        use_proj = n_filters != projection_dim

        if use_highway or use_proj:
            embedding = tf.reshape(embedding, [-1, n_filters])
            if self.bidirectional:
                embedding_reverse = tf.reshape(embedding_reverse,
                                               [-1, n_filters])

        # set up weights for projection
        if use_proj:
            assert n_filters > projection_dim
            with tf.variable_scope('CNN_proj') as scope:
                W_proj_cnn = tf.get_variable(
                    "W_proj", [n_filters, projection_dim],
                    initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
                    dtype=DTYPE)
                b_proj_cnn = tf.get_variable(
                    "b_proj", [projection_dim],
                    initializer=tf.constant_initializer(0.0),
                    dtype=DTYPE)

        # apply highways layers
        def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
            carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
            transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
            return carry_gate * transform_gate + (1.0 - carry_gate) * x

        if use_highway:
            highway_dim = n_filters

            for i in range(n_highway):
                with tf.variable_scope('CNN_high_%s' % i) as scope:
                    W_carry = tf.get_variable(
                        'W_carry', [highway_dim, highway_dim],
                        # glorit init
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_carry = tf.get_variable(
                        'b_carry', [highway_dim],
                        initializer=tf.constant_initializer(-2.0),
                        dtype=DTYPE)
                    W_transform = tf.get_variable(
                        'W_transform', [highway_dim, highway_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_transform = tf.get_variable(
                        'b_transform', [highway_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=DTYPE)

                embedding = high(embedding, W_carry, b_carry,
                                 W_transform, b_transform)
                if self.bidirectional:
                    embedding_reverse = high(embedding_reverse,
                                             W_carry, b_carry,
                                             W_transform, b_transform)
                self.token_embedding_layers.append(
                    tf.reshape(embedding,
                               [batch_size, unroll_steps, highway_dim])
                )

        # finally project down to projection dim if needed
        if use_proj:
            embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn
            if self.bidirectional:
                embedding_reverse = tf.matmul(embedding_reverse, W_proj_cnn) \
                                    + b_proj_cnn
            self.token_embedding_layers.append(
                tf.reshape(embedding,
                           [batch_size, unroll_steps, projection_dim])
            )

        # reshape back to (batch_size, tokens, dim)
        if use_highway or use_proj:
            shp = [batch_size, unroll_steps, projection_dim]
            embedding = tf.reshape(embedding, shp)
            if self.bidirectional:
                embedding_reverse = tf.reshape(embedding_reverse, shp)

        # at last assign attributes for remainder of the model
        self.embedding = embedding
        if self.bidirectional:
            self.embedding_reverse = embedding_reverse

    # def _build(self):
    #     # size of input options
    #     n_tokens_vocab = self.options['n_tokens_vocab']
    #     batch_size = self.options['batch_size']
    #     unroll_steps = self.options['unroll_steps']
    #
    #     # LSTM options
    #     lstm_dim = self.options['lstm']['dim']
    #     projection_dim = self.options['lstm']['projection_dim']
    #     n_lstm_layers = self.options['lstm'].get('n_layers', 1)
    #     dropout = self.options['dropout']
    #     keep_prob = 1.0 - dropout
    #
    #     if self.char_inputs:
    #         self._build_word_char_embeddings()
    #     else:
    #         self._build_word_embeddings()
    #
    #     # now the LSTMs
    #     # these will collect the initial states for the forward
    #     #   (and reverse LSTMs if we are doing bidirectional)
    #     self.init_lstm_state = []
    #     self.final_lstm_state = []
    #
    #     # get the LSTM inputs
    #     # self.embedding.shape = (batch_size, unroll_steps, words dim)
    #     if self.bidirectional:
    #         lstm_inputs = [self.embedding, self.embedding_reverse]
    #     else:
    #         lstm_inputs = [self.embedding]
    #
    #     # now compute the LSTM outputs
    #     cell_clip = self.options['lstm'].get('cell_clip')
    #     proj_clip = self.options['lstm'].get('proj_clip')
    #
    #     use_skip_connections = self.options['lstm'].get(
    #         'use_skip_connections')
    #     if use_skip_connections:
    #         print("USING SKIP CONNECTIONS")
    #
    #     lstm_outputs = []
    #     # ############# #
    #     #    biLSTM     #
    #     # ############# #
    #     # The code use two LSTM to implement the bilstm
    #     # lstm_num: for bidirectional, there are two complete lstm system,
    #     # eg. forward lstm: X-->lstm-->lstm-->...-->last lstm
    #     #     backward lstm: X-->lstm-->lstm-->...-->last lstm
    #     #     forward lstm-->softmax
    #     #     backward lstm --> softmax
    #     for lstm_num, lstm_input in enumerate(lstm_inputs):
    #         lstm_cells = []
    #         for i in range(n_lstm_layers):
    #             if projection_dim < lstm_dim:
    #                 # are projecting down output
    #                 lstm_cell = tf.nn.rnn_cell.LSTMCell(
    #                     lstm_dim, num_proj=projection_dim,
    #                     cell_clip=cell_clip, proj_clip=proj_clip, name='layer%d'%i)
    #             else:
    #                 lstm_cell = tf.nn.rnn_cell.LSTMCell(
    #                     lstm_dim,
    #                     cell_clip=cell_clip, proj_clip=proj_clip, name='layer%d'%i)
    #
    #             if use_skip_connections:
    #                 # ResidualWrapper adds inputs to outputs
    #                 if i == 0:
    #                     # don't add skip connection from token embedding to
    #                     # 1st layer output
    #                     pass
    #                 else:
    #                     # add a skip connection
    #                     lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)
    #
    #             # add dropout
    #             if self.is_training:
    #                 lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
    #                                                           input_keep_prob=keep_prob)
    #
    #             lstm_cells.append(lstm_cell)
    #
    #         if n_lstm_layers > 1:
    #             lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
    #         else:
    #             lstm_cell = lstm_cells[0]
    #
    #         with tf.control_dependencies([lstm_input]):
    #             self.init_lstm_state.append(
    #                 lstm_cell.zero_state(batch_size, DTYPE))
    #             # NOTE: this variable scope is for backward compatibility
    #             # with existing models...
    #             # TODO: need to add sequence length to the lstm
    #             if lstm_num == 0:
    #                 scope_name = 'bilstm/fw'
    #             else:
    #                 scope_name = 'bilstm/bw'
    #             _lstm_output_unpacked, final_state = tf.nn.static_rnn(
    #                 lstm_cell,
    #                 tf.unstack(lstm_input, axis=1),
    #                 initial_state=self.init_lstm_state[-1],
    #             scope=scope_name)
    #             self.final_lstm_state.append(final_state)
    #
    #
    #         # (batch_size * unroll_steps, 512)
    #         lstm_output_flat = tf.reshape(
    #             tf.stack(_lstm_output_unpacked, axis=1), [-1, projection_dim])
    #
    #         if self.is_training:
    #             # add dropout to output
    #             lstm_output_flat = tf.nn.dropout(lstm_output_flat,
    #                                              keep_prob)
    #         tf.add_to_collection('lstm_output_embeddings',
    #                              _lstm_output_unpacked)
    #
    #         lstm_outputs.append(lstm_output_flat)
    #
    #     self._build_loss(lstm_outputs)

    def _build(self):
        # size of input options
        n_tokens_vocab = self.options['n_tokens_vocab']
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        # LSTM options
        lstm_dim = self.options['lstm']['dim']
        projection_dim = self.options['lstm']['projection_dim']
        n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        dropout = self.options['dropout']
        keep_prob = 1.0 - dropout

        if self.char_inputs:
            self._build_word_char_embeddings()
        else:
            self._build_word_embeddings()

        # now the LSTMs
        # these will collect the initial states for the forward
        #   (and reverse LSTMs if we are doing bidirectional)
        self.init_lstm_state = []
        self.final_lstm_state = []

        # get the LSTM inputs
        # self.embedding.shape = (batch_size, unroll_steps, words dim)
        if self.bidirectional:
            lstm_inputs = [self.embedding, self.embedding_reverse]
        else:
            lstm_inputs = [self.embedding]

        # now compute the LSTM outputs
        cell_clip = self.options['lstm'].get('cell_clip')
        proj_clip = self.options['lstm'].get('proj_clip')

        use_skip_connections = self.options['lstm'].get(
            'use_skip_connections')
        if use_skip_connections:
            print("USING SKIP CONNECTIONS")

        lstm_outputs = []
        # ############# #
        #    biLSTM     #
        # ############# #
        # The code use two LSTM to implement the bilstm
        # lstm_num: for bidirectional, there are two complete lstm system,
        # eg. forward lstm: X-->lstm-->lstm-->...-->last lstm
        #     backward lstm: X-->lstm-->lstm-->...-->last lstm
        #     forward lstm-->softmax
        #     backward lstm --> softmax
        for lstm_num, lstm_input in enumerate(lstm_inputs):
            outputs = lstm_input
            for i in range(n_lstm_layers):
                if projection_dim < lstm_dim:
                    # are projecting down output
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim, num_proj=projection_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip, name='layer%d'%i)
                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip, name='layer%d'%i)

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

                self.init_lstm_state.append(
                    lstm_cell.zero_state(batch_size, DTYPE))
                # NOTE: this variable scope is for backward compatibility
                # with existing models...
                # TODO: need to add sequence length to the lstm
                if lstm_num == 0:
                    scope_name = 'bilstm/fw'
                else:
                    scope_name = 'bilstm/bw'
                _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                    lstm_cell,
                    tf.unstack(outputs, axis=1),
                    initial_state=self.init_lstm_state[-1],
                scope=scope_name)
                self.final_lstm_state.append(final_state)
                outputs = tf.stack(_lstm_output_unpacked, axis=1)

            # (batch_size * unroll_steps, 512)
            lstm_output_flat = tf.reshape(
                outputs, [-1, projection_dim])

            if self.is_training:
                # add dropout to output
                lstm_output_flat = tf.nn.dropout(lstm_output_flat,
                                                 keep_prob)
            tf.add_to_collection('lstm_output_embeddings',
                                 _lstm_output_unpacked)

            lstm_outputs.append(lstm_output_flat)

        self._build_loss(lstm_outputs)

    # This step build the model
    def _build_loss(self, lstm_outputs):
        '''
        Create:
            self.total_loss: total loss op for training
            self.softmax_W, softmax_b: the softmax variables
            self.next_token_id / _reverse: placeholders for gold input

        '''
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        n_tokens_vocab = self.options['n_tokens_vocab']

        # DEFINE next_token_id and *_reverse placeholders for the gold input
        def _get_next_token_placeholders(suffix):
            name = 'next_token_id' + suffix
            id_placeholder = tf.placeholder(DTYPE_INT,
                                            shape=(batch_size, unroll_steps),
                                            name=name)
            return id_placeholder

        # get the window and weight placeholders
        self.next_token_id = _get_next_token_placeholders('')
        if self.bidirectional:
            self.next_token_id_reverse = _get_next_token_placeholders(
                '_reverse')

        # DEFINE THE SOFTMAX VARIABLES
        # get the dimension of the softmax weights
        # softmax dimension is the size of the output projection_dim
        softmax_dim = self.options['lstm']['projection_dim']

        # the output softmax variables -- they are shared if bidirectional
        if self.share_embedding_softmax:
            # softmax_W is just the embedding layer
            self.softmax_W = self.embedding_weights

        with tf.variable_scope('softmax'), tf.device('/cpu:0'):
            # Glorit init (std=(1.0 / sqrt(fan_in))
            softmax_init = tf.random_normal_initializer(0.0,
                                                        1.0 / np.sqrt(softmax_dim))
            if not self.share_embedding_softmax:
                self.softmax_W = tf.get_variable(
                    'W', [n_tokens_vocab, softmax_dim],
                    dtype=DTYPE,
                    initializer=softmax_init
                )
            self.softmax_b = tf.get_variable(
                'b', [n_tokens_vocab],
                dtype=DTYPE,
                initializer=tf.constant_initializer(0.0))

        # now calculate losses
        # loss for each direction of the LSTM
        self.individual_losses = []

        if self.bidirectional:
            next_ids = [self.next_token_id, self.next_token_id_reverse]
        else:
            next_ids = [self.next_token_id]

        for id_placeholder, lstm_output_flat in zip(next_ids, lstm_outputs):
            # flatten the LSTM output and next token id gold to shape:
            # (batch_size * unroll_steps, softmax_dim)
            # Flatten and reshape the token_id placeholders
            next_token_id_flat = tf.reshape(id_placeholder, [-1, 1])

            with tf.control_dependencies([lstm_output_flat]):
                if self.is_training and self.sample_softmax:
                    losses = tf.nn.sampled_softmax_loss(
                        self.softmax_W, self.softmax_b,
                        next_token_id_flat, lstm_output_flat,
                        self.options['n_negative_samples_batch'],
                        self.options['n_tokens_vocab'],
                        num_true=1)

                else:
                    # get the full softmax loss
                    output_scores = tf.matmul(
                        lstm_output_flat,
                        tf.transpose(self.softmax_W)
                    ) + self.softmax_b
                    # NOTE: tf.nn.sparse_softmax_cross_entropy_with_logits
                    #   expects unnormalized output since it performs the
                    #   softmax internally
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=output_scores,
                        labels=tf.squeeze(next_token_id_flat, squeeze_dims=[1])
                    )

            self.individual_losses.append(tf.reduce_mean(losses))

        # now make the total loss -- it's the mean of the individual losses
        if self.bidirectional:
            self.total_loss = 0.5 * (self.individual_losses[0]
                                     + self.individual_losses[1])
        else:
            self.total_loss = self.individual_losses[0]

    def weight_layers(self, name, bilm_ops, l2_coef=None,
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
        # lm_embeddings.shape = (max sentence length, bilstm layers, bilstm output layer dims)
        lm_embeddings = bilm_ops['lm_embeddings']
        mask = bilm_ops['mask']

        n_lm_layers = int(lm_embeddings.get_shape()[1])
        lm_dim = int(lm_embeddings.get_shape()[3])

        with tf.control_dependencies([lm_embeddings, mask]):
            # Cast the mask and broadcast for layer use.
            mask_float = tf.cast(mask, 'float32')
            broadcast_mask = tf.expand_dims(mask_float, axis=-1)

            def _do_ln(x):
                # do layer normalization excluding the mask
                x_masked = x * broadcast_mask
                N = tf.reduce_sum(mask_float) * lm_dim
                mean = tf.reduce_sum(x_masked) / N
                variance = tf.reduce_sum(((x_masked - mean) * broadcast_mask) ** 2
                                         ) / N
                return tf.nn.batch_normalization(
                    x, mean, variance, None, None, 1E-12
                )

            if use_top_only:
                layers = tf.split(lm_embeddings, n_lm_layers, axis=1)
                # just the top layer
                sum_pieces = tf.squeeze(layers[-1], squeeze_dims=1)
                # no regularization
                reg = 0.0
            else:
                W = tf.get_variable(
                    '{}_ELMo_W'.format(name),
                    shape=(n_lm_layers,),
                    initializer=tf.zeros_initializer,
                    regularizer=_l2_regularizer,
                    trainable=True,
                )

                # normalize the weights
                normed_weights = tf.split(
                    tf.nn.softmax(W + 1.0 / n_lm_layers), n_lm_layers
                )
                # split LM layers
                layers = tf.split(lm_embeddings, n_lm_layers, axis=1)

                # compute the weighted, normalized LM activations
                pieces = []
                for w, t in zip(normed_weights, layers):
                    if do_layer_norm:
                        pieces.append(w * _do_ln(tf.squeeze(t, squeeze_dims=1)))
                    else:
                        pieces.append(w * tf.squeeze(t, squeeze_dims=1))
                sum_pieces = tf.add_n(pieces)

                # get the regularizer
                reg = [
                    r for r in tf.get_collection(
                        tf.GraphKeys.REGULARIZATION_LOSSES)
                    if r.name.find('{}_ELMo_W/'.format(name)) >= 0
                ]
                if len(reg) != 1:
                    raise ValueError

            # scale the weighted sum by gamma
            gamma = tf.get_variable(
                '{}_ELMo_gamma'.format(name),
                shape=(1,),
                initializer=tf.ones_initializer,
                regularizer=None,
                trainable=True,
            )
            weighted_lm_layers = sum_pieces * gamma

            ret = {'weighted_op': weighted_lm_layers, 'regularization_op': reg}

        return ret




