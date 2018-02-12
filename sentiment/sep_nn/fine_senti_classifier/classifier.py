import tensorflow as tf
import numpy as np
from sentiment.util.coarse.senti_data_generator import DataGenerator
from sentiment.util.coarse.metrics import  Metrics


class SentiFunction:
    def __init__(self, nn_config):
        self.nn_config = nn_config

    # sentiment expression: for sentiment towards specific attribute
    # the sentiment is choosen by attribute
    def sentiment_matrix(self, graph):
        W = tf.get_variable(name='senti_mat', initializer=tf.random_uniform(shape=(
            self.nn_config['normal_senti_prototype_num'] * 3 + self.nn_config['attribute_senti_prototype_num'] *
            self.nn_config['attributes_num'],
            self.nn_config['sentiment_dim']), dtype='float32'))
        graph.add_to_collection('W', W)
        return W

    def sentiment_attention(self, h, W, m, graph):
        """
        :param h: is a sentence. (number of words, lstm cell size)
        :param W: sentiment expression for sentence x; shape = (number of sentiment prototypes, lstm cell size)
        :param m: mask to eliminate influence of 0
        :return: shape = (number of words, number of sentiment prototypes)
        """

        temp = tf.multiply(m, tf.exp(tf.matmul(h, W, transpose_b=True)))
        # denominator of softmax
        den = tf.reduce_sum(temp, axis=1, keep_dims=True)
        den = tf.tile(den, multiples=[1, self.nn_config['normal_senti_prototype_num'] * 3 + self.nn_config[
            'attribute_senti_prototype_num'] * self.nn_config['attributes_num']])
        attention = tf.truediv(temp, den)
        graph.add_to_collection('senti_attention', attention)
        return attention

    def attended_sentiment(self, W, attention, graph):
        """

        :param W: one (yi,ai)
        :param attention: (number of words, number of prototypes)
        :param graph: 
        :return: w.shape=(number of words in a sentence,sentiment_dim) 
        """
        attention = tf.expand_dims(attention, axis=2)
        # attention.shape = (number of words, number of sentiment prototypes, sentiment dim)
        attention = tf.tile(attention, multiples=[1, 1, self.nn_config['sentiment_dim']])
        w = tf.reduce_sum(tf.multiply(attention, W), axis=1)
        graph.add_to_collection('w', w)
        return w

    # association between attribute and sentiment: towards specific attribute
    def attribute_distribution(self, A, h, graph):
        """
        distribution of all attributes in this sentence
        :param A: A.shape = (attributes number +1 , attributes dim(=lstm cell size)) or 
                  A.shape = (number of words, number of attributes+1, attribute dim(=lstm cell size))
        :param h: one sentence
        :param graph: 
        :return: shape = (number of attributes+1, number of words)
        """
        if not self.nn_config['is_mat']:
            # A.shape=(number of attributes+1, attribute dim(=word dim))
            # A_dist = (number of attributes+1,number of words)
            A_dist = tf.nn.softmax(tf.matmul(A, h, transpose_b=True))
        else:
            print('is_mat')
            # A.shape = (number of words, number of attributes+1, attribute dim(=lstm cell dim))
            # new_A.shape = (number of words, number of attributes+1)
            new_A = []
            for i in range(self.nn_config['words_num']):
                word_embed = h[i]
                # wi_A.shape = (number of attributes+1, attribute dim(=lstm cell dim))
                wi_A = A[i]
                new_A.append(tf.reduce_sum(tf.multiply(wi_A, word_embed), axis=1))
            print('out is mat')
            A_dist = tf.nn.softmax(tf.transpose(new_A))
        graph.add_to_collection('attribute_distribution', A_dist)
        return A_dist

    def relative_pos_matrix(self, graph):
        V = tf.get_variable(name='relative_pos',
                            initializer=tf.random_uniform(shape=(self.nn_config['rps_num'], self.nn_config['rp_dim']),
                                                          dtype='float32'))
        graph.add_to_collection('V', V)
        return V

    def vi(self, i, a_dist, V, graph):
        """
        :param i: the position of hi
        :param a_dist: shape = (number of words, ); one attribute's distribution in a sentence.
        :param V: shape = (number of relative position, relative position dim)
        :return: vi of attribute a at position i. shape = (relative position dim,)
        """
        a_dist = tf.expand_dims(a_dist, axis=1)
        a_dist = tf.tile(a_dist, multiples=[1, self.nn_config['rp_dim']])
        vi = []
        for k in range(self.nn_config['words_num']):
            ak = a_dist[k]
            if abs(k - i) < self.nn_config['rps_num']:
                v_r = V[abs(k - i)]
            else:
                last = self.nn_config['rps_num'] - 1
                v_r = V[last]
            vi.append(tf.multiply(ak, v_r))
        vi = tf.reduce_sum(vi, axis=0)
        return vi

    def Vi(self, A_dist, V, graph):
        """

        :param A_dist: shape = (number of attributes+1, number of words)
        :param V: shape = (number of relative position, relative position dim)
        :param graph: 
        :return: realtive position vector of each attribute at each position.
                shape = (number of attributes+1, number of words, relative position dim)
        """
        # A_vi.shape=(attributes number+1, number of words, relative position dim)
        A_vi = []
        for i in range(self.nn_config['attributes_num']+1):
            a_dist = A_dist[i]
            # a_vi.shape=(number of words, relative position dim)
            a_vi = []
            for j in range(self.nn_config['words_num']):
                # v.shape=(relative position dim,)
                v = self.vi(j, a_dist, V, graph)
                a_vi.append(v)
            A_vi.append(a_vi)
        graph.add_to_collection('A_vi', A_vi)
        return A_vi

    def beta(self, graph):
        """

        :param graph: 
        :return: beta weight, shape=(rp_dim)
        """
        b = tf.get_variable(name='beta',initializer=tf.random_uniform(shape=(self.nn_config['rp_dim'],), dtype='float32'))
        graph.add_to_collection('beta', b)
        return b

    # sentiment score
    def score(self, item1, item2, graph):
        """
        :param item1: shape = (3*number of attributes+3, number of words in a sentence)
        :param item2: shape = (number of attributes+1, number of words in a sentence)
        :param graph: 
        :return: (3*number of attributes+1,) this is all combinations of yi and ai
        """
        # item2.shape = (3*self.nn_config['attributes_num'],self.nn_config['words_num'])
        item2 = tf.reshape(tf.tile(tf.expand_dims(item2, axis=1), [1, 3, 1]),
                           shape=(3 * self.nn_config['attributes_num']+3, self.nn_config['words_num']))



        score = tf.reduce_max(tf.add(item1, item2), axis=1)
        graph.add_to_collection('senti_score', score)
        return score

    def max_f_senti_score(self, senti_label, score, graph):
        """

        :param senti_label: shape=(attributes numbers, 3) 
        :param score: shape=(attributes numbers, 3)
        :param graph: 
        :return: shape = (number of attributes,3)
        """
        # if value is 1 then it is true, otherwise flase
        condition = tf.equal(tf.ones_like(senti_label, dtype='float32'), senti_label)
        # max_score.shape = (number of attributes, 1)
        max_fscore = tf.reduce_max(tf.where(condition,
                                            tf.ones_like(score, dtype='float32') * tf.constant(-np.inf, dtype='float32'),
                                            score),
                                   axis=1, keep_dims=True)
        # consider when attribute contains all sentiment in a sentence.
        max_fscore = tf.where(tf.is_inf(max_fscore), tf.zeros_like(max_fscore, dtype='float32'), max_fscore)
        max_fscore = tf.tile(max_fscore, multiples=[1, 3])
        graph.add_to_collection('max_f_senti_score', max_fscore)
        return max_fscore

    def senti_loss_mask(self, atr_label, senti_label):
        """
        calculate mask for sentiment loss. a 0 position is determined by three elements: prediction of attribute score,
        attribute labels and sentiment labels. if the attribute is not detected, not a true label or a sentiment is false, 
        then its sentiment loss ly should be zero.
        :param atr_label: shape = (number of attributes+1, )
        :param senti_label: shape = (number of attributes, 3)
        :return: 
        """
        atr_label = tf.tile(tf.expand_dims(atr_label, axis=1), multiples=[1, 3])
        return atr_label * senti_label

    def loss(self, senti_label, score, atr_label, graph):
        """
        shape of loss = (sentiment)
        :param senti_label: shape=(attributes numbers+1, 3) the second part is one-hot to represent which sentiment it is.
        :param score: shape=(3*attributes numbers+3,)
        :param atr_label: shape = (attribute numbers+1,)
        :param graph:
        :return: loss for a sentence for all true attributes and mask all false attributes.
        """
        score = tf.reshape(score, shape=(self.nn_config['attributes_num']+1, 3))
        # max_f_score.shape = (number of attributes+1, 3)
        max_f_score = self.max_f_senti_score(senti_label, score, graph)
        theta = tf.constant(self.nn_config['sentiment_loss_theta'], dtype='float32')
        # loss shape = (number of attributes+1, 3)
        senti_loss = tf.add(tf.subtract(theta, score), max_f_score)
        # mask.shape=(number of attributes+1,3)
        mask = self.senti_loss_mask(atr_label, senti_label)
        masked_loss = tf.multiply(mask, senti_loss)
        final_loss = tf.reduce_sum(masked_loss)
        graph.add_to_collection('senti_loss', final_loss)
        return final_loss

    def prediction(self,score,atr_label,graph):
        """
        :param score: shape = (3*attributes numbers+3,)
        :param atr_label: shape = (attributes numbers+1,)
        :param graph: 
        :return: 
        """
        score = tf.reshape(score, shape=(self.nn_config['attributes_num'], 3))
        atr_label = tf.tile(tf.expand_dims(atr_label, axis=1), multiples=[1, 3])
        # atr_label.shape = (attributes numbers, 3)
        score = tf.multiply(atr_label,score)
        condition = tf.greater(score,self.nn_config['senti_pred_threshold'])
        pred = tf.where(condition,tf.ones_like(score,dtype='float32'),tf.zeros_like(score,dtype='float32'))
        graph.add_to_collection('prediction',pred)
        return pred

    # ===============================================
    # ============= attribute function ==============
    # ===============================================
    def attribute_vec(self, graph):
        """
        
        :param graph: 
        :return: shape = (number of attributes+1, attributes dim)
        """
        # A is matrix of attribute vector
        A = []
        for i in range(self.nn_config['attributes_num']):
            att_vec = tf.get_variable(name='att_vec' + str(i),
                                      initializer=tf.random_uniform(shape=(self.nn_config['attribute_dim'],),
                                                                    dtype='float32'))
            A.append(att_vec)
        graph.add_to_collection('A', A)
        o = tf.get_variable(name='other_vec', initializer=tf.random_uniform(shape=(self.nn_config['attribute_dim'],),
                                                                            dtype='float32'))
        graph.add_to_collection('o', o)
        A= tf.concat([A,tf.expand_dims(o,axis=0)],axis=0)
        return A

    def attribute_mat(self, graph):
        """
        
        :param graph: 
        :return: shape = (attributes number+1, attribute mat size, attribute dim)
        """
        # all attribute mention matrix are the same, but some a padded with 0 vector. We use mask to avoid the training process
        A_mat = []
        for i in range(self.nn_config['attributes_num']):
            att_mat = tf.get_variable(name='att_mat' + str(i),
                                      initializer=tf.random_uniform(shape=(self.nn_config['attribute_mat_size'],
                                                                           self.nn_config['attribute_dim']),
                                                                    dtype='float32'))
            A_mat.append(att_mat)
        graph.add_to_collection('A_mat', A_mat)
        o_mat = tf.get_variable(name='other_vec',
                                initializer=tf.random_uniform(shape=(self.nn_config['attribute_mat_size'],
                                                                     self.nn_config['attribute_dim']),
                                                              dtype='float32'))
        graph.add_to_collection('o_mat', o_mat)

        A_mat = tf.concat([A_mat,tf.expand_dims(o_mat,axis=0)],axis=0)

        return A_mat

    def attribute_mat_attention(self, att_mat, word_embed, graph):
        """
        attribute attetion for one attribute and one word
        :param att_mat: (attribute_mat_size,attribute dim); attribute dim = lstm cell size
        :param word_embed: shape = (lstm cell size,)
        :param graph: 
        :return: shape=(attribute matrix size, attribute dim),  attribute dim is the same to lstm cell dim
        """
        attention = tf.nn.softmax(tf.reduce_sum(tf.multiply(att_mat, word_embed), axis=1))
        attention = tf.expand_dims(attention, axis=1)
        attention = tf.tile(attention, multiples=[1, self.nn_config['attribute_dim']])
        graph.add_to_collection('att_mat_attention', attention)
        return attention

    def attribute_mat2vec(self, word_embed, A_mat, graph):
        """
        
        :param word_embed: shape = (lstm cell size, )
        :param A_mat: (number of attributes, number of attribute mention prototypes,attribute dim)
        :param graph: 
        :return: shape = (number of attributes, attribute dim)
        """
        # A is matrix attribute matrix
        A = []
        for att_mat in A_mat:
            attention = self.attribute_mat_attention(att_mat, word_embed, graph)
            att_vec = tf.reduce_sum(tf.multiply(attention, att_mat), axis=0)
            A.append(att_vec)
        graph.add_to_collection('A', A)
        return A

    def words_attribute_mat2vec(self, x, A_mat, graph):
        """
        convert attribtes matrix to attributes vector for each words in a sentence. A_mat include non-attribute mention matrix.
        :param x: 
        :param A_mat: 
        :param o_mat: 
        :param graph: 
        :return: shape = (number of words, number of attributes, attribute dim(=lstm cell dim))
        """
        words_A = []
        for i in range(self.nn_config['words_num']):
            word_embed = x[i]
            A= self.attribute_mat2vec(word_embed, A_mat, graph)
            words_A.append(A)
        graph.add_to_collection('words_attributes', words_A)
        return words_A


class Classifier:
    def __init__(self, nn_config):
        self.nn_config = nn_config
        self.sf = SentiFunction(nn_config)
        self.dg = DataGenerator(nn_config)

    def sentences_input(self, graph):
        X = tf.placeholder(
            shape=(self.nn_config['batch_size'], self.nn_config['words_num']),
            dtype='float32')
        graph.add_to_collection('X', X)
        return X

    def attribute_labels_input(self, graph):
        """
        
        :param graph: 
        :return: shape = (batch size, attributes number)
        """
        y_att = tf.placeholder(shape=(self.nn_config['batch_size'], self.nn_config['attributes_num']+1), dtype='float32')
        graph.add_to_collection('y_att', y_att)
        return y_att

    def sentiment_labels_input(self, graph):
        """
        :param graph: 
        :return: shape=[batch_size, number of attributes, 3], thus ys=[...,sentence[...,attj_senti[0,1,0],...],...]
        """
        y_senti = tf.placeholder(shape=(self.nn_config['batch_size'], self.nn_config['attributes_num']+1, 3),dtype='float32')
        graph.add_to_collection('y_senti', y_senti)
        return y_senti

    # should use variable share
    def sentence_lstm(self, X,graph):
        """
        return a lstm of a sentence
        :param x: a sentence
        :param graph: 
        :return: 
        """
        weight = tf.get_variable(name='sentence_lstm_w',
                                 initializer=tf.random_uniform(shape=(self.nn_config['word_dim'],
                                                                      self.nn_config['lstm_cell_size']),
                                                               dtype='float32'))
        bias = tf.get_variable(name='sentence_lstm_b',
                               initializer=tf.zeros(shape=(self.nn_config['lstm_cell_size']), dtype='float32'))
        tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(weight)

        X = tf.reshape(X, shape=(-1, self.nn_config['word_dim']))
        Xt = tf.add(tf.matmul(X, weight), bias)
        Xt = tf.reshape(Xt, shape=(-1, self.nn_config['words_num'], self.nn_config['lstm_cell_size']))
        # xt = tf.add(tf.expand_dims(tf.matmul(x, weight), axis=0), bias)
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size'])
        init_state = cell.zero_state(batch_size=self.nn_config['batch_size'], dtype='float32')
        # outputs.shape = (batch size, max_time, cell size)
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs=Xt, initial_state=init_state, time_major=False)
        graph.add_to_collection('sentence_lstm_outputs', outputs)
        return outputs

    def senti_extors_mat(self, graph):
        """
        input a matrix to extract sentiment expression for attributes in sentences. The last one extract sentiment expression for non-attribute.
        The non-attribute only has one sentiment: NEU
        shape of the extractors matrix: [3*attribute numbers+1, 
                              normal sentiment prototype numbers*3 + attributes sentiment prototypes number*attribute number,
                              sentiment epression dim(sentiment dim)]
        :param graph: 
        :return: 
        """
        extors = self.sentiment_extract_mat()
        extors = tf.constant(extors, dtype='float32')
        graph.add_to_collection('senti_extractor', extors)
        return extors



    # @ normal_function
    def extor_expandNtile(self, extor, proto_num):
        """

        :param exter: it is extractor  
        :return: 
        """
        extor = np.expand_dims(extor, axis=1)
        extor = np.tile(extor, reps=[1, self.nn_config[proto_num]])
        extor = np.expand_dims(extor, axis=2)
        extor = np.tile(extor, reps=[1, 1, self.nn_config['sentiment_dim']])
        extor = np.reshape(extor, (-1, self.nn_config['sentiment_dim']))
        return extor

    # @ normal_function
    def sentiment_extract_mat(self):
        """
        This function return a matrix to extract expression prototype for all (yi,ai) combinations.
        :return: [ sentiment extractor for one sentence[...,[1,...],...,[0,...],...] ,...] 
        """
        # label = np.ones(shape=(self.nn_config['attributes_num'],), dtype='float32')
        extors = []
        for i in range(self.nn_config['attributes_num']):
            # attribute sentiment prototypes
            att_senti_ext = np.zeros(shape=(self.nn_config['attributes_num'],), dtype='float32')
            att_senti_ext[i] = 1
            att_senti_ext = self.extor_expandNtile(att_senti_ext, proto_num='attribute_senti_prototype_num')
            for j in range(3):
                # normal sentiment prototypes
                normal_senti_ext = np.zeros(shape=(3,), dtype='float32')
                normal_senti_ext[j] = 1
                normal_senti_ext = self.extor_expandNtile(normal_senti_ext, proto_num='normal_senti_prototype_num')
                extors.append(np.concatenate([normal_senti_ext, att_senti_ext], axis=0))
        for i in range(3):
            # non-attribute sentiment prototypes
            o_att_senti_ext = np.zeros(shape=(self.nn_config['attributes_num']),dtype='float32')
            o_att_senti_ext = self.extor_expandNtile(o_att_senti_ext,'attribute_senti_prototype_num')
            # non-attribute normal sentiment prototypes
            o_normal_senti_ext = np.zeros(shape=(3,),dtype='float32')
            o_normal_senti_ext[i] = 1
            o_normal_senti_ext = self.extor_expandNtile(o_normal_senti_ext, proto_num='normal_senti_prototype_num')
            extors.append(np.concatenate([o_normal_senti_ext,o_att_senti_ext],axis=0))
        return np.array(extors)

    def extors_mask(self, extors, graph):
        """
        when calculate p(w|h), need to eliminate the the influence of false sentiment.
        :param extor: shape = (3*attribute numbers +3, 
                               self.nn_config['normal_senti_prototype_num'] * 3 +
                               self.nn_config['attribute_senti_prototype_num'] * self.nn_config['attributes_num'],
                               sentiment dim)
        :param graph: 
        :return: (3*attributes number+3, number of sentiment expression prototypes)
        """
        extors = tf.reduce_sum(extors,axis=2)
        condition = tf.equal(extors,np.zeros_like(extors,dtype='float32'))
        mask = tf.where(condition,tf.zeros_like(extors,dtype='float32'),tf.ones_like(extors,dtype='float32'))
        graph.add_to_collection('extors_mask',mask)
        return mask

    def optimizer(self, losses, graph):
        opt = tf.train.AdamOptimizer(self.nn_config['lr']).minimize(tf.reduce_mean(losses))
        graph.add_to_collection('opt', opt)
        return opt

    def lookup_table(self, X, mask, graph):
        """
        :param X: shape = (batch_size, words numbers)
        :param mask: used to prevent update of #PAD#
        :return: shape = (batch_size, words numbers, word dim)
        """
        table = tf.placeholder(shape=(2074276, self.nn_config['word_dim']), dtype='float32')
        graph.add_to_collection('table', table)
        table = tf.Variable(table, name='table')

        embeddings = tf.nn.embedding_lookup(table, X, partition_strategy='mod', name='lookup_table')
        embeddings = tf.multiply(embeddings,mask)
        graph.add_to_collection('lookup_table', embeddings)
        return embeddings

    def is_word_padding_input(self,X,graph):
        """
        To make the sentence have the same length, we need to pad each sentence with '#PAD#'. To avoid updating of the vector,
        we need a mask to multiply the result of lookup table.
        :param graph: 
        :return: shape = (review number, sentence number, words number)
        """
        ones = tf.ones_like(X, dtype='int32')*2074275
        is_one = tf.equal(X, ones)
        mask = tf.where(is_one, tf.zeros_like(X, dtype='float32'), tf.ones_like(X, dtype='float32'))
        mask = tf.tile(tf.expand_dims(mask, axis=2), multiples=[1,1,self.nn_config['word_dim']])
        return mask

    def classifier(self):
        graph = tf.Graph()
        with graph.as_default():
            X = self.sentences_input(graph=graph)
            words_pad_M = self.is_word_padding_input(X, graph)
            X = self.lookup_table(X, words_pad_M,graph)
            # lstm
            with tf.variable_scope('sentence_lstm'):
                # H.shape = (batch size, max_time, cell size)
                H = self.sentence_lstm(X, graph=graph)

            y_att = self.attribute_labels_input(graph=graph)
            y_senti = self.sentiment_labels_input(graph=graph)
            if not self.nn_config['is_mat']:
                A = self.sf.attribute_vec(graph)
            else:
                A = self.sf.attribute_mat(graph)
            # sentiment expression prototypes matrix
            # shape = (3*numbers of normal sentiment prototype + attributes_numbers*attribute specific sentiment prototypes)
            W = self.sf.sentiment_matrix(graph)
            # sentiment extractors for all (yi,ai)
            # extors_mat.shape = (3*attributes number+3, sentiment prototypes number, sentiment dim)
            extors_mat = self.senti_extors_mat(graph)
            # extors_mask_mat.shape = (3*attributes number+3, sentiment prototypes number)
            extors_mask_mat = self.extors_mask(extors=extors_mat,graph=graph)
            # relative position matrix
            V = self.sf.relative_pos_matrix(graph)
            beta = self.sf.beta(graph)
            # extract sentiment expression corresponding to sentiment and attribute from W for all attributes
            # W.shape=(number of attributes*3+3, size of original W); shape of original W =(3*normal sentiment prototypes + attribute number * attribute sentiment prototypes, sentiment dim)
            W = tf.multiply(extors_mat, W)
        losses = []
        predictions = []
        for i in range(self.nn_config['batch_size']):
            with graph.as_default():
                # x.shape=(words number, word dim)
                x = X[i]
                h = H[i]
                atr_label = y_att[i]
                senti_label = y_senti[i]
                # if the attribute is represented by a mat, we need to convert it to a vector based on a word
                if self.nn_config['is_mat']:
                    # A.shape = (number of words, number of attributes+1, attribute dim(=lstm cell dim))
                    A = self.sf.words_attribute_mat2vec(x=h, A_mat=A, graph=graph)
                    # A = []
                    # for l in range(len(words_A_o)):
                    #     A.append(words_A_o[l][0])
                    #     o.append(words_A_o[l][1])

            # sentiment function
            with graph.as_default():

                # item1 shape (3*number of attributes+3, number of words in a sentence)
                item1 = []
                for j in range(3 * self.nn_config['attributes_num']+3):
                    # attention.shape=(number of words, number of sentiment prototypes)
                    attention = self.sf.sentiment_attention(h, W[j], extors_mask_mat[j], graph)
                    # w.shape =(number of words in sentence, sentiment dim); w represents sentiment expression for
                    # each words in sentence
                    w = self.sf.attended_sentiment(W[j], attention, graph)
                    item1.append(tf.reduce_sum(tf.multiply(w, h), axis=1))
                # all attributes distribution for one sentence
                # A.shape = (number of words, number of attributes+1, attribute dim(=lstm cell dim))
                # or A.shape = (number of attributes+1, attribute dim)
                A_dist = self.sf.attribute_distribution(A=A, h=h, graph=graph)
                # A_dist.shape = (number of attributes+1, number of words)
                A_vi = self.sf.Vi(A_dist=A_dist, V=V, graph=graph)
                # item2.shape=(number of attributes+1, number of words in a sentence)
                item2 = tf.reduce_sum(tf.multiply(A_vi, beta), axis=2)
                # senti_socre.shape = (3*number of attributes+1,)
                senti_score = self.sf.score(item1, item2, graph)
                # sentim_loss.shape = (number of attributes+1, 3)
                senti_loss = self.sf.loss(senti_label, senti_score, atr_label, graph)
                losses.append(senti_loss)
                # prediction
                senti_pred = self.sf.prediction(score=senti_score,atr_label=atr_label,graph=graph)
                predictions.append(senti_pred)

        with graph.as_default():
            opt = self.optimizer(losses=losses, graph=graph)
            saver = tf.train.Saver()
        return graph, saver

    def train(self):
        graph, saver = self.classifier()
        with graph.as_default():
            # input
            X = graph.get_collection('X')
            # labels
            y_att = graph.get_collection('y_att')
            y_senti = graph.get_collection('y_senti')
            # train_step
            train_step = graph.get_collection('train_step')
            # attribute function
            init = tf.global_variables_initializer()
        with graph.device('/gpu:1'):
            with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                sess.run(init)
                for i in range(self.nn_config['epoch']):
                    sentences, att_labels, senti_labels = self.dg.gen(i) + 3
                    senti_extors = self.sentiment_extract_mat()
                    sess.run(init)
                    sess.run(train_step, feed_dict={X: sentences, y_att: att_labels, y_senti: senti_labels})