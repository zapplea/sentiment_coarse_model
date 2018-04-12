import tensorflow as tf

class RelScore:
    def __init__(self,nn_config):
        self.nn_config = nn_config

    def aspect_probability_input(self,graph):
        """
        Input the probability of each aspect
        :param graph: (batch size, attributes num)
        :return: 
        """
        aspect_prob = tf.placeholder(name='aspect_prob',shape=(None,self.nn_config['attributes_num']),dtype='float32')
        return aspect_prob

    def sentence_lstm(self, X, seq_len, graph):
        """
        return a lstm of a sentence
        :param X: shape = (batch size * max review length, words number, word dim)
        :param seq_len: shape = (batch,) show the number of words in a batch
        :param graph: 
        :return: outputs.shape = (batch size * max review length, words number, word dim);
                 last_hidden.shape = (batch size * max review length, word dim)
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

    def relevance_prob_atr(self, atr_score, graph):
        """
        :param atr_score: (batch size, max review length, attributes num)
        :return: shape = (batch size, max review length, attributes num) , in dimension 2 values are the same
        """
        # prob.shape = (batch size, attributes num, max review length); p(x;a)
        rel_prob = tf.nn.softmax(tf.transpose(atr_score,perm=[0,2,1]),axis=2)
        # prob.shape = (batch size,max review length, attributes num)
        rel_prob = tf.transpose(rel_prob,perm=[0,2,1])
        return rel_prob

    def aspect_prob(self,aspect_prob,graph):
        """
        
        :param aspect_prob: shape=(batch size, attributes num)
        :param graph: 
        :return: (batch size, max review length, attributes num)
        """
        # aspect_prob.shape = (batch size, max review length, attributes num)
        aspect_prob = tf.tile(tf.expand_dims(aspect_prob,axis=1),multiples=[1,self.nn_config['max_review_length'],1])
        return aspect_prob

    def score(self,aspect_prob,rel_prob,atr_score):
        """
        
        :param aspect_prob: (batch size, max review length, attributes num)
        :param rel_prob: (batch size, max review length, attributes num)
        :param atr_score: (batch size, max review length, attributes num)
        :return: 
        """
        score = tf.multiply(rel_prob,tf.multiply(aspect_prob,atr_score))
        return score

    def relevance_prob_senti(self,H):
        pass