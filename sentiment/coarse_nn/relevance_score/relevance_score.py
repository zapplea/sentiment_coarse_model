import tensorflow as tf

class RelScore:
    def __init__(self,nn_config):
        self.nn_config = nn_config

    def sentence_lstm(self, X, seq_len, graph):
        """
        return a lstm of a sentence
        :param X: shape = (batch size, words number, word dim)
        :param seq_len: shape = (batch,) show the number of words in a batch
        :param graph: 
        :return: ()
        """
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size'])
        # outputs.shape = (batch size, max_time, cell size)
        outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=X, time_major=False, sequence_length=seq_len, dtype='float32')
        graph.add_to_collection('sentence_lstm_outputs', outputs)
        # get the last lifted embedding for each sentence
        index0 = tf.cast(tf.expand_dims(tf.range(start=0,
                                                 limit=self.nn_config['max_sentences_num']*self.nn_config['batch_size']),
                                        axis=1),
                         dtype='int32')
        index1 = tf.cast(tf.expand_dims(tf.expand_dims(seq_len-1,axis=1)),dtype='int32')
        slice_index = tf.concat([index0,index1],axis=1)
        last_hidden = tf.gather(outputs,indices=slice_index)
        return outputs, last_hidden

    def review2sentence(self,review, feature_dim):
        """
        
        :param review: shape = (batch size, max reviews length,) 
        :return: 
        """
        sentences = tf.reshape(review,shape = (-1,self.nn_config['words_num'],feature_dim))
        return sentences

    def sentence2review(self,sentences, feature_dim):
        reviews = tf.reshape(sentences,(-1,self.nn_config['max_review_length'],self.nn_config['words_num'],feature_dim))
        return reviews

    def relevance_score_atr(self, h):
        """
        
        :param h: shape=(batch size * max review length, words num, word dim)
        :return: 
        """
        W = tf.get_variable(name='',shape=,)
        bias =

    def relevance_score_senti(self,H):
        pass