import tensorflow as tf
import numpy as np

class Initializer:
    @staticmethod
    def parameter_initializer(shape,dtype='float32'):
        stdv=1/tf.sqrt(tf.constant(shape[-1],dtype=dtype))
        init = tf.random_uniform(shape,minval=-stdv,maxval=stdv,dtype=dtype,seed=1)
        return init

class AttributeFunction:
    def __init__(self, nn_config):
        self.nn_config = nn_config
        self.initializer=Initializer.parameter_initializer

    def attribute_mat(self, reg, graph):
        """

        :param graph: 
        :return: shape = (attributes number+1, attribute mat size, attribute dim)
        """
        A_mat = tf.get_variable(name='A_mat',initializer=self.initializer(shape=(self.nn_config['attributes_num'],
                                                                    self.nn_config['attribute_mat_size'],
                                                                    self.nn_config['attribute_dim']),
                                                            dtype='float32'))
        reg['attr_reg'].append(tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(A_mat))
        o_mat = tf.get_variable(name='o_mat',initializer=self.initializer(shape=(1,
                                                                   self.nn_config['attribute_mat_size'],
                                                                   self.nn_config['attribute_dim']),
                                                            dtype='float32'))
        reg['attr_reg'].append(tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(o_mat))
        return A_mat,o_mat

    def words_attribute_mat2vec(self, H, A_mat, graph):
        """
        convert attribtes matrix to attributes vector for each words in a sentence. A_mat include non-attribute mention matrix.
        :param H: shape = (batch size, number of words, word dim)
        :param A_mat: (number of atr, atr mat size, atr dim)
        :param graph: 
        :return: shape = (batch size, number of words, number of attributes, attribute dim(=lstm cell dim))
        """
        # H.shape = (batch size, words number, attribute number, word dim)
        H = tf.tile(tf.expand_dims(H, axis=2), multiples=[1, 1, self.nn_config['attributes_num'], 1])
        # H.shape = (batch size, words number, attribute number, attribute mat size, word dim)
        H = tf.tile(tf.expand_dims(H, axis=3), multiples=[1, 1, 1, self.nn_config['attribute_mat_size'], 1])
        # attention.shape = (batch size, words number, attribute number, attribute mat size)
        attention = tf.nn.softmax(tf.reduce_sum(tf.multiply(H, A_mat), axis=4))
        # attention.shape = (batch size, words number, attribute number, attribute mat size, attribute dim)
        attention = tf.tile(tf.expand_dims(attention, axis=4), multiples=[1, 1, 1, 1, self.nn_config['attribute_dim']])
        words_A = tf.reduce_sum(tf.multiply(attention, A_mat), axis=3)
        return words_A

    def words_nonattribute_mat2vec(self, H, o_mat, graph):
        """

        :param H: shape = (batch size, words number, word dim)
        :param o_mat: shape = (1,attribute mat size, attribute dim)
        :param graph: 
        :return: batch size, number of words, attributes num, attribute dim( =word dim)
        """
        # H.shape = (batch size, words number, 1, word dim)
        H = tf.expand_dims(H, axis=2)
        # H.shape = (batch size, words number, 1, attribute mat size, word dim)
        H = tf.tile(tf.expand_dims(H, axis=3), multiples=[1, 1, 1, self.nn_config['attribute_mat_size'], 1])
        # attention.shape = (batch size, words number, 1, attribute mat size)
        attention = tf.nn.softmax(tf.reduce_sum(tf.multiply(H, o_mat), axis=4))
        # attention.shape = (batch size, words number, 1, attribute mat size, attribute dim)
        attention = tf.tile(tf.expand_dims(attention, axis=4), multiples=[1, 1, 1, 1, self.nn_config['attribute_dim']])
        # words_A.shape = (batch size, number of words, 1, attribute dim( =word dim))
        words_o = tf.reduce_sum(tf.multiply(attention, o_mat), axis=3)
        # words_A.shape = (batch size, number of words, attributes number, attribute dim( =word dim))
        words_o = tf.tile(words_o, multiples=[1, 1, self.nn_config['attributes_num'], 1])
        return words_o

    def sentence_score(self, A, X, mask, graph):
        """

        :param A: shape = (number of attributes, attribute dim) or
                  shape = (batch size, words number, attributes num, attribute dim)
        :param X: shape = (batch size, words number, lstm cell size)
        :param graph: 
        :return: (batch size, attributes num, words num)
        """
        # TODO: should eliminate the influence of #PAD# when calculate reduce max
        # X.shape = (batch size, words num, attributes num, attribute dim)
        X = tf.tile(tf.expand_dims(X, axis=2), multiples=[1, 1, self.nn_config['attributes_num'], 1])
        # score.shape = (batch size, words num, attributes num)
        score = tf.reduce_sum(tf.multiply(A, X), axis=3)
        # score.shape = (batch size, attributes num, words num)
        score = tf.transpose(score, [0, 2, 1])
        # mask.shape = (batch size, attributes number, words num)
        mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, self.nn_config['attributes_num'],1])
        score = tf.add(score, mask)
        # score.shape = (batch size, attributes num)
        # score = tf.reduce_max(score, axis=2)
        return score

    def sentence_sigmoid(self,score,graph):
        """
        
        :param score: shape=(batch size*max review length, attributes num)
        :param graph: 
        :return: 
        """
        return tf.nn.sigmoid(score)

    def review_mask(self,X, graph):
        """
        mask padded sentence in review_sigmoid()
        :param X: shape=(batch size, max review length, wrods num)
        :param graph: 
        :return:  shape = (batch size*max review length, coarse attributes num)
        """
        paddings = tf.ones_like(X, dtype='int32') * self.nn_config['padding_word_index']
        condition = tf.equal(paddings, X)
        condition = tf.reduce_all(condition, axis=2)
        #shape = (batch size, max review length)
        mask = tf.where(condition, tf.zeros_like(condition, dtype='float32'), tf.ones_like(condition, dtype='float32'))

        mask = tf.reshape(tf.tile(tf.expand_dims(mask,axis=2),multiples=[1,1,self.nn_config['coarse_attributes_num']]),
                          shape=(-1,self.nn_config['coarse_attributes_num']))
        return mask

    def review_score(self,sentence_prob,mask,review_len,reg,graph):
        """
        
        :param sentence_prob: shape = (batch size * max review length, attributes num)
        :param mask: shape = (batch size*max review length, coarse attributes num)
        :param review_len: # shape = (batch size,)
        :param graph: 
        :return: (batch size, coarse_attributes_num)
        """
        W = tf.get_variable(name='W_trans',
                            initializer=self.initializer(shape=(self.nn_config['coarse_attributes_num'],
                                                                self.nn_config['attributes_num']),
                                                         dtype='float32'))
        reg['attr_reg'].append(tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W))
        # shape = (batch size * max review length, coarse_attributes_num, attributes num)
        sentence_prob = tf.tile(tf.expand_dims(sentence_prob,axis=1),multiples=[1,self.nn_config['coarse_attributes_num'],1])
        # shape = (batch size * max review length, coarse_attributes_num)
        sentence_prob = tf.reduce_sum(tf.multiply(W,sentence_prob),axis=2)
        # shape = (batch size * max review length, coarse_attributes_num)
        sentence_prob = tf.multiply(sentence_prob,mask)
        # shape = (batch size, max review length, coarse_attributes_num)
        sentence_prob = tf.reshape(sentence_prob,shape = (-1,self.nn_config['max_review_len'],self.nn_config['coarse_attributes_num']))
        # shape = (batch size, coarse attributes num)
        review_len = tf.tile(tf.expand_dims(review_len,axis=1),multiples=[1,self.nn_config['coarse_attributes_num']])
        # shape = (batch size, coarse_attributes_num)
        review_score = tf.truediv(tf.reduce_sum(sentence_prob,axis=1),review_len)

        return review_score

    def sigmoid_loss(self, name, score, Y_att, reg_list,graph):
        """

        :param score: shape=(batch size*max review length, attributes num)
        :return: 
        """
        loss = tf.reduce_mean(tf.add(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_att, logits=score),
                                                   axis=1),
                                     tf.reduce_sum(reg_list)))
        graph.add_to_collection(name,loss)
        return loss

    def prediction(self,name, score, graph):
        """
        :param score: shape = (batch size, coarse attributes num) 
        :param graph: 
        :return: 
        """
        #
        prob = tf.sigmoid(score)
        condition = tf.greater(prob,self.nn_config['review_atr_pred_threshold'])
        pred = tf.where(condition, tf.ones_like(score, dtype='float32'), tf.zeros_like(score, dtype='float32'))
        graph.add_to_collection(name, pred)
        return pred


class SentimentFunction:
    def __init__(self,nn_config):
        self.nn_config = nn_config
        self.initializer = Initializer.parameter_initializer

    def sentiment_matrix(self,reg,graph):
        W = tf.get_variable(name='senti_mat', initializer=self.initializer(shape=(
            self.nn_config['normal_senti_prototype_num'] * self.nn_config['sentiment_num'] + self.nn_config['attribute_senti_prototype_num'] *
            self.nn_config['attributes_num'],
            self.nn_config['sentiment_dim']), dtype='float32'))
        reg['senti_reg'].append(tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W))
        return W

    def senti_extors_mat(self,graph):
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
        return extors

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
            for j in range(self.nn_config['sentiment_num']):
                # normal sentiment prototypes
                normal_senti_ext = np.zeros(shape=(self.nn_config['sentiment_num'],), dtype='float32')
                normal_senti_ext[j] = 1
                normal_senti_ext = self.extor_expandNtile(normal_senti_ext, proto_num='normal_senti_prototype_num')
                extors.append(np.concatenate([normal_senti_ext, att_senti_ext], axis=0))
        for i in range(self.nn_config['sentiment_num']):
            # non-attribute sentiment prototypes
            o_att_senti_ext = np.zeros(shape=(self.nn_config['attributes_num']), dtype='float32')
            o_att_senti_ext = self.extor_expandNtile(o_att_senti_ext, 'attribute_senti_prototype_num')
            # non-attribute normal sentiment prototypes
            o_normal_senti_ext = np.zeros(shape=(self.nn_config['sentiment_num'],), dtype='float32')
            o_normal_senti_ext[i] = 1
            o_normal_senti_ext = self.extor_expandNtile(o_normal_senti_ext, proto_num='normal_senti_prototype_num')
            extors.append(np.concatenate([o_normal_senti_ext, o_att_senti_ext], axis=0))
        return np.array(extors)

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
        extors = tf.reduce_sum(extors, axis=2)
        condition = tf.equal(extors, np.zeros_like(extors, dtype='float32'))
        mask = tf.where(condition, tf.zeros_like(extors, dtype='float32'), tf.ones_like(extors, dtype='float32'))
        return mask

    def relative_pos_matrix(self, reg, graph):
        V = tf.get_variable(name='relative_pos',
                            initializer=self.initializer(shape=(self.nn_config['rps_num'], self.nn_config['rps_dim']),
                                                          dtype='float32'))
        reg['senti_reg'].append(tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(V))
        return V

    def relative_pos_ids(self, graph):
        """
        :param graph: 
        :return: shape = (number of words, number of words)
        """
        id4sentence = []
        for i in range(self.nn_config['words_num']):
            id4word_i = []
            for j in range(self.nn_config['words_num']):
                if abs(i - j) < self.nn_config['rps_num']:
                    id4word_i.append(abs(i - j))
                else:
                    id4word_i.append(self.nn_config['rps_num'] - 1)
            id4sentence.append(id4word_i)
        rp_ids = tf.constant(id4sentence, dtype='int32')
        return rp_ids

    def beta(self, reg, graph):
        """

        :param graph: 
        :return: beta weight, shape=(rp_dim)
        """
        b = tf.get_variable(name='beta',
                            initializer=self.initializer(shape=(self.nn_config['rps_dim'],), dtype='float32'))
        reg['senti_reg'].append(tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(b))
        return b

    def sentiment_attention(self, H, W, mask, graph):
        """
        :param H: shape = (batch size, number of words, lstm cell size)
        :param W: shape = (3*attribute numbers + 3,number of sentiment prototypes, lstm cell size). 3*attribute numbers is
        3 sentiment for each attributes; 3 is sentiment for non-attribute entity, it only has normal sentiment, not attribute
        specific sentiment.
        :param mask: mask to eliminate influence of 0; (3*attributes number+3, number of sentiment expression prototypes)
        :return: shape = (batch size,number of words, 3+3*attributes number, number of sentiment prototypes).
        """
        # # H.shape = (batch size, words num, 3+3*attributes number, word dim)
        # H = tf.tile(tf.expand_dims(H, axis=2), multiples=[1, 1, self.nn_config['sentiment_num'] * self.nn_config[
        #     'attributes_num'] + self.nn_config['sentiment_num'], 1])
        # # H.shape = (batch size, words num, 3+3*attributes number, sentiment prototypes, word dim)
        # H = tf.tile(tf.expand_dims(H, axis=3), multiples=[1, 1, 1, self.nn_config['normal_senti_prototype_num'] * self.nn_config['sentiment_num'] +
        #                                                   self.nn_config['attribute_senti_prototype_num'] *
        #                                                   self.nn_config['attributes_num'],
        #                                                   1])
        # temp.shape = (batch size, words num, 3+3*attributes number, sentiment prototypes num)
        temp = tf.multiply(mask, tf.exp(tf.tensordot(H, W,axes=[[-1],[-1]])))

        # denominator.shape = (batch size, words num, 3+3*attributes number, 1)
        denominator = tf.reduce_sum(temp, axis=3, keepdims=True)

        denominator = tf.tile(denominator, multiples=[1, 1, 1,
                                                      self.nn_config['normal_senti_prototype_num'] * self.nn_config['sentiment_num'] +
                                                      self.nn_config['attribute_senti_prototype_num'] * self.nn_config[
                                                          'attributes_num']])
        attention = tf.truediv(temp, denominator)
        return attention

    def attended_sentiment(self, W, attention, graph):
        """
        :param W: all (yi,ai); shape = (3*number of attribute +3, sentiment prototypes, sentiment dim)
        :param attention: shape = (batch size, number of words, 3+3*attributes number, number of sentiment prototypes)
        :param graph: 
        :return: (batch size,number of words, 3+3*attributes number, sentiment dim)
        """
        # # attention.shape = (batch size, number of words, 3+3*attributes number, number of sentiment prototypes, sentiment dim)
        # attention = tf.tile(tf.expand_dims(attention, axis=4), multiples=[1, 1, 1, 1, self.nn_config['sentiment_dim']])
        # # (batch size,number of words, 3+3*attributes number, sentiment dim)
        # attended_W = tf.reduce_sum(tf.multiply(attention, W), axis=3)

        # shape of each scalar in attention splits = (batch size, number of words, 1, number of sentiment prototypes)
        attention_splits = tf.split(attention,
                                    num_or_size_splits=self.nn_config['sentiment_num']*(1+self.nn_config['attributes_num']),
                                    axis=2)
        # shape of each scalar in W splits = (1, sentiment prototypes, sentiment dim)
        W_splits = tf.split(W,
                            num_or_size_splits=self.nn_config['sentiment_num']*(1+self.nn_config['attributes_num']),
                            axis=0)
        # (3*attribute num+3, batch size ,number of words, 1, sentiment dim)
        attented_W_ls=[]
        for a, w in zip(attention_splits,W_splits):
            # (batch size ,number of words, sentiment dim) --> (batch size ,number of words, 1, sentiment dim)
            attented_W_ls.append(tf.expand_dims(tf.tensordot(a,w,axes=[[-2,-1],[0,1]]),axis=2))
        # (batch size,number of words, 3+3*attributes number, sentiment dim)
        attended_W = tf.concat(attented_W_ls,axis=2)

        return attended_W

    def item1(self, W, H, graph):
        """

        :param W: shape = (batch size,number of words, 3+3*attributes number, sentiment dim)
        :param H: shape = (batch size, number of words, word dim)
        :return: shape = (batch size,number of words, 3+3*attributes number)
        """
        # H.shape = (batch size,number of words, 3+3*attributes number, sentiment dim)
        H = tf.tile(tf.expand_dims(H, axis=2), multiples=[1, 1, self.nn_config['sentiment_num'] * self.nn_config['attributes_num'] + self.nn_config['sentiment_num'], 1])
        item1_score = tf.reduce_sum(tf.multiply(W, H), axis=3)
        return item1_score

    def words_attribute_mat2vec(self, H, A_mat, graph):
        """
        convert attribtes matrix to attributes vector for each words in a sentence. A_mat include non-attribute mention matrix.
        :param H: shape = (batch size, number of words, word dim)
        :param A_mat: (number of atr, atr mat size, atr dim)
        :param graph: 
        :return: shape = (batch size, number of words, number of attributes + 1, attribute dim(=lstm cell dim))
        """
        # H.shape = (batch size, words number, attribute number+1, word dim)
        H = tf.tile(tf.expand_dims(H, axis=2), multiples=[1, 1, self.nn_config['attributes_num'] + 1, 1])
        # H.shape = (batch size, words number, attribute number+1, attribute mat size, word dim)
        H = tf.tile(tf.expand_dims(H, axis=3), multiples=[1, 1, 1, self.nn_config['attribute_mat_size'], 1])
        # attention.shape = (batch size, words number, attribute number, attribute mat size)
        attention = tf.nn.softmax(tf.reduce_sum(tf.multiply(H, A_mat), axis=4))
        # attention.shape = (batch size, words number, attribute number, attribute mat size, attribute dim)
        attention = tf.tile(tf.expand_dims(attention, axis=4), multiples=[1, 1, 1, 1, self.nn_config['attribute_dim']])
        words_A = tf.reduce_sum(tf.multiply(attention, A_mat), axis=3)
        return words_A

    # association between attribute and sentiment: towards specific attribute
    def attribute_distribution(self, A, H, graph):
        """
        distribution of all attributes in this sentence
        :param A: A.shape = (attributes number +1 , attributes dim(=lstm cell size)) or 
                  A.shape = (batch size, number of words, number of attributes + 1, attribute dim(=lstm cell dim))
        :param H: batch size, words num, word dim
        :param graph: 
        :return: shape = (batch size, number of attributes+1, wrods number)
        """
        if not self.nn_config['is_mat']:
            H = tf.reshape(H, shape=(-1, self.nn_config['lstm_cell_size']))
            # A.shape=(number of attributes+1, attribute dim(=lstm cell size))
            # A_dist = (batch size,number of attributes+1,number of words)
            A_dist = tf.nn.softmax(tf.transpose(tf.reshape(tf.matmul(A, H, transpose_b=True),
                                                           shape=(self.nn_config['attributes_num'] + 1, -1,
                                                                  self.nn_config['words_num'])), [1, 0, 2]))
        else:
            # A.shape = (batch size, number of words, number of attributes+1, attribute dim(=lstm cell dim))
            # H.shape = (batch size, number of words, number of attributes+1, word dim)

            H = tf.tile(tf.expand_dims(H, axis=2), multiples=[1, 1, self.nn_config['attributes_num'] + 1, 1])
            # A_dist.shape = (batch size, attributes number, words number)
            A_dist = tf.nn.softmax(tf.transpose(tf.reduce_sum(tf.multiply(A, H), axis=3), [0, 2, 1]))
        return A_dist

    def rd_Vi(self, A_dist, V, rp_ids, graph):
        """
        :param A_dist: shape = (batch size, number of attributes+1, number of words)
        :param V: shape = (number of relative position, relative position dim)
        :param rp_ids: shape = (number of words, number of words)
        :param graph:
        :return: realtive position vector of each attribute at each position.
        shape = (batch size, number of attributes+1, number of words, relative position dim)
        """
        # rp_mat.shape = (number of words, number of words, rp_dim)
        rp_mat=tf.nn.embedding_lookup(V,rp_ids)
        # A_dist.shape = (batch size, number of attributes+1, number of words,relative position dim)
        A_dist = tf.tile(tf.expand_dims(A_dist,axis=3),multiples=[1,1,1,self.nn_config['rps_dim']])
        # A_dist.shape = (batch size, number of attributes+1, number of words, number of words,relative position dim)
        A_dist = tf.tile(tf.expand_dims(A_dist,axis=2),multiples=[1,1,self.nn_config['words_num'],1,1])
        # A_Vi.shape = (batch size, number of attributes+1, number of words, relative position dim)
        A_Vi = tf.reduce_sum(tf.multiply(A_dist,rp_mat),axis=3)
        return A_Vi

    def mask_for_pad_in_score(self, X, graph):
        """
        This mask is used in score, to eliminate the influence of pad words when reduce_max. This this mask need to add to the score.
        Since 0*inf = nan
        :param X: the value is word id. shape=(batch size, max words num)
        :param graph: 
        :return: 
        """
        paddings = tf.ones_like(X, dtype='int32') * self.nn_config['padding_word_index']
        condition = tf.equal(paddings, X)
        mask = tf.where(condition, tf.ones_like(X, dtype='float32') * tf.convert_to_tensor(-np.inf),
                        tf.zeros_like(X, dtype='float32'))
        mask = tf.reshape(mask, shape=[-1, self.nn_config['words_num']])
        return mask

    # sentiment score
    def score(self, item1, item2, mask, graph):
        """
        :param item1: shape = (batch size,number of words, 3+3*attributes number)
        :param item2: shape=(batch size, number of attributes+1, number of words)
        :param graph: 
        :return: (batch size, 3+3*attributes number, number of words) this is all combinations of yi and ai
        """
        # item1.shape = (batch size, 3+3*attributes number, number of words)
        item1 = tf.transpose(item1, [0, 2, 1])
        # item2.shape = (batch size, 3+3*attributes number, number of words)
        item2 = tf.reshape(tf.tile(tf.expand_dims(item2, axis=2), [1, 1, 3, 1]),
                           shape=(-1, self.nn_config['sentiment_num'] * self.nn_config['attributes_num'] + self.nn_config['sentiment_num'], self.nn_config['words_num']))
        # score.shape = (batch size, 3+3*attributes number, number of words)
        score = tf.add(item1, item2)
        # mask.shape = (batch size, attributes number, words num)
        mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, self.nn_config['sentiment_num'] + self.nn_config['sentiment_num'] * self.nn_config['attributes_num'], 1])
        # eliminate influence of #PAD#
        score = tf.add(score, mask)

        return score

    def coarse_score(self,fine_score, reg, graph):
        """
        
        :param fine_score: shape = (batch size*max review length, number of attributes+1,3)
        :return: sahpe = (batch size, coarse attr num + 1, 3)
        """
        with tf.variable_scope('coarse_senti',reuse=tf.AUTO_REUSE):
            # shape = (fine attr num*3+3, coarse attr num*3+3)
            W = tf.get_variable(name='fine2coarse',
                                initializer=self.initializer(shape=(self.nn_config['attributes_num']*self.nn_config['sentiment_num']+self.nn_config['sentiment_num'], self.nn_config['coarse_attributes_num']*self.nn_config['sentiment_num']+self.nn_config['sentiment_num']), dtype='float32'))
            reg['senti_reg'].append(tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W))
        # shape = (batch size * max review length, attr num*3 + 3)
        fine_score = tf.reshape(fine_score,shape=(-1, self.nn_config['attributes_num']*self.nn_config['sentiment_num']+self.nn_config['sentiment_num']))
        # shape = (batch size * max review length, coarse attr num*3 + 3)
        coarse_score= tf.matmul(fine_score,W)
        # shape = (batch size, max review length, coarse attr num*3 + 3)
        coarse_score = tf.reshape(coarse_score,shape=(-1,self.nn_config['max_review_len'],self.nn_config['coarse_attributes_num']*self.nn_config['sentiment_num']+self.nn_config['sentiment_num']))
        # sahpe = (batch size, coarse attr num*3 + 3)
        coarse_score = tf.reduce_sum(coarse_score,axis=1)
        return tf.reshape(coarse_score,shape=(-1,self.nn_config['coarse_attributes_num']+1,self.nn_config['sentiment_num']))


    def softmax_loss(self, name, labels, logits, reg_list, graph):
        """

        :param labels: (batch size, number of attributes+1,3)
        :param logits: (batch_size, number of attributes + 1, 3)
        :return: 
        """

        loss = tf.reduce_mean(tf.add(
            tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, dim=-1), axis=1),
            tf.reduce_sum(reg_list)))
        graph.add_to_collection(name, loss)
        return loss

    def expand_attr_labels(self, labels, graph):
        """
        The attribute model only detect attribute, and will not show none-attribute obviously. So, in here need to add none-attribute,
        based on the result of the attribute model.
        :param graph: 
        :param labels: shape = (batch size, attributes number)
        :return: shape = (batch size, attributes number+1)
        """

        Y_att = labels
        # TODO: add non-attribute
        batch_size = tf.shape(Y_att)[0]
        non_attr = tf.zeros((batch_size, 1), dtype='float32')
        # condition = tf.equal(tf.reduce_sum(Y_att, axis=1, keepdims=True), non_attr)
        # non_attr = tf.where(condition, tf.ones_like(non_attr), non_attr)
        Y_att = tf.concat([Y_att, non_attr], axis=1)
        return Y_att

    def prediction(self,name, score, Y_att, graph):
        """
        :param score: shape = (batch size, attributes numbers+1,3)
        :param Y_att: shape = (batch size, attributes numbers+1)
        :param graph: 
        :return: 
        """
        # score.shape = (batch size, attributes numbers+1,3)
        score = tf.nn.softmax(logits=score,axis=-1)
        # pred.shape =(batch size, attributes number +1 , 3)
        pred = tf.where(tf.equal(tf.reduce_max(score,axis=2,keep_dims=True),score),tf.ones_like(score),tf.zeros_like(score))
        # use Y_atr to mask non-activated attributes' sentiment
        Y_att = tf.tile(tf.expand_dims(Y_att,axis=2),multiples=[1,1,self.nn_config['sentiment_num']])
        pred = tf.multiply(Y_att, pred)
        graph.add_to_collection(name, pred)
        return pred
