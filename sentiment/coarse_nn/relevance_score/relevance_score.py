import tensorflow as tf

class RelScore:
    def __init__(self,nn_config):
        self.nn_config = nn_config

    def sentence_lstm(self, X, seq_len, graph):
        """
        return a lstm of a sentence
        :param X: shape = (batch size, words number, word dim)
        :param seq_len: shape = (batch size,) show the number of words in a batch
        :param graph: 
        :return: 
        """
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size'])
        # outputs.shape = (batch size, max_time, cell size)
        outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=X, time_major=False, sequence_length=seq_len, dtype='float32')
        graph.add_to_collection('sentence_lstm_outputs', outputs)
        # get the last lifted embedding for each sentence
        index0 = tf.range(start=0,limit=self.nn_config['max_sentences_num']*self.nn_config['batch_size'])

        return outputs

    def relevance_score(self, H):
        pass