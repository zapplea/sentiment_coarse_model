import tensorflow as tf
import jieba
import pickle
import gensim
from sklearn.utils import check_array


class DataGenerator:
    def __init__(self,data_config):
        self.data_config = data_config
        self.reviews = self.read_reviews()
        self.tiled_reviews = self.tile_reviews()
        self.word2num,self.embed_mat = self.read_wordembedding()
        self.encoded_reviews = self.encode_review()

    def read_reviews(self):
        # convert a review to fixed number of sentences
        # [ [[s1,s2,...],aspect],...] ==> [[s1,...,sk],...] [aspect1,...]
        # si=[w1,...]
        # aspect=[w1,...]
        with open(self.data_config['reviews_f'],'rb') as f:
            reviews = pickle.load(f)
        return reviews

    def tile_reviews(self):
        tiled_reviews=[]
        for cur_review in self.reviews:
            cur_aspect = cur_review[1]
            cur_sentences = cur_review[0]
            tiled_aspect=[]
            for i in range(self.data_config['aspect_words_num']):
                if i<len(cur_aspect):
                    tiled_aspect.append(cur_aspect[i])
                else:
                    tiled_aspect.append('#PAD#')
            tiled_sentence_list=[]
            for i in range(self.data_config['sentences_num']):
                if i < len(cur_sentences):
                    cur_sentence = cur_sentences[i]
                    tiled_sentence=[]
                    for j in range(self.data_config['sentence_words_num']):
                        if j<len(cur_sentence):
                            tiled_sentence.append(cur_sentence[j])
                        else:
                            tiled_sentence.append('#PAD#')
                    tiled_sentence_list.append(tiled_sentence)
                else:
                    tiled_sentence_list.append(['#PAD#']*self.data_config['sentence_words_num'])
            tiled_reviews.append([tiled_sentence_list,tiled_aspect])
        return tiled_reviews

    def read_wordembedding(self):
        word_embed=gensim.models.Word2Vec.load(self.data_config['wordembedding_f'])
        # word dictionary
        vocabulary=word_embed.index2word # word is stored in order of word_embed
        word2num={}
        for i in range(len(vocabulary)):
            word2num[vocabulary[i]]=i
        # word embedding
        embed_mat=word_embed.syn0
        embed_mat = check_array(embed_mat, dtype='float32', order='C')
        return word2num,embed_mat

    def encode_review(self):
        encoded_reviews=[]
        for review in self.tiled_reviews:
            aspect=review[1]
            sentence_list=review[0]
            encoded_aspect=[]
            for word in aspect:
                code=self.word2num[word]
                encoded_aspect.append(code)
            encoded_sentence_list=[]
            for sentence in sentence_list:
                encoded_sentence=[]
                for word in sentence:
                    encoded_sentence.append(self.word2num[word])
                encoded_sentence_list.append(encoded_sentence)
            encoded_reviews.append([encoded_sentence_list,encoded_aspect])
        return encoded_reviews

    def gen_data(self,epoch,mode):
        test_data = self.encoded_reviews[-300:]
        train_data = self.encoded_reviews[:-300]
        train_data_size = len(train_data)
        if mode == 'test':
            return test_data
        else:
            start = epoch * self.data_config['batch_size'] % train_data_size
            end = (epoch * self.data_config['batch_size']  + self.data_config['batch_size'] ) % train_data_size
            if start < end:
                batches = train_data[start:end]
            elif start >= end:
                batches = train_data[start:]
                # if mode=="train":
                batches.extend(train_data[0:end])
            return batches


class Classifier:
    def __init__(self,rnn_config):
        self.rnn_config=rnn_config

    def input(self,graph):
        X = tf.placeholder(shape=self.rnn_config['shape']['input'],dtype='float32')
        graph.add_to_collection(name='input',value=X)
        return X

    def aspect(self,graph):
        # [a1,a2,...,an]
        A = tf.placeholder(shape=self.rnn_config['shape']['aspect'],dtype='float32')
        graph.add_to_collection(name='aspect',value=A)
        return A

    def ground_truth(self,graph):
        Y_ = tf.placeholder(shape=self.rnn_config['shape']['gt'],dtype='float32')
        graph.add_to_collection(name='ground_truth',value=Y_)
        return Y_

    def aspect_rnn(self,a,graph):
        # [[w,...],[w,...],...,[w,...]] ==> [a1,a2,...,ak]
        weight = tf.get_variable(name=self.rnn_config['name']['aspect_rnn'] + '_w',
                                 initializer=tf.random_uniform(shape=self.rnn_config['shape']['aspect_rnn']),
                                 dtype='float32')
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.rnn_config['lamda'])(weight))
        bias = tf.get_variable(name=self.rnn_config['name']['aspect_rnn'] + '_b',
                               initializer=tf.ones(shape=self.rnn_config['shape']['aspect_rnn'][1], dtype='float32'),
                               dtype='float32')
        n_steps = self.rnn_config['aspect_rnn_n_steps']
        cell_size = self.rnn_config['aspect_rnn_cell_size']
        A = tf.reshape(a, shape=(-1, self.rnn_config['word_dim']))
        At = tf.reshape(tf.matmul(A, weight) + bias, shape=(-1, n_steps, cell_size))
        cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size)
        init_state = cell.zero_state(self.rnn_config['batch_size'], dtype='float32')
        outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=At, initial_state=init_state)
        outputs=tf.transpose(outputs,[1,0,2])
        graph.add_to_collection('aspect_rnn_output',outputs[-1])
        return outputs[-1]

    def aspect_transform(self,A,graph):
        temp=tf.expand_dims(A, 1)
        temp=tf.expand_dims(temp,0)
        A = tf.tile(temp,multiples=[self.rnn_config['sentence_num'],1,1,1])
        graph.add_to_collection('aspect_transformed',A)
        return A

    def attention(self,x,a,graph):
        """
        :param x: sentences, [[[w1,w2,...,wm],[w1,w2,..,wm],...],...] 
        :param a: aspect, [[[a1],[a2],...[ak]], ..., [[a1],[a2],...,[ak]]]
        :param graph: 
        :return: 
        """
        a=self.aspect_transform(a,graph)
        aspect=tf.tile(a,multiples=[1,1,self.rnn_config['sentence_words_num'],1])
        att_factor=tf.concat([x,aspect],axis=3)
        graph.add_to_collection('att_factor',att_factor)
        weight= tf.get_variable(name=self.rnn_config['name']['attention']+'_w',
                                initializer=tf.random_uniform(
                                    shape=self.rnn_config['shape']['att_linear_weight'],
                                    dtype='float32'),
                                dtype='float32')
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.rnn_config['lamda'])(weight))
        bias=tf.get_variable(name=self.rnn_config['name']['attention']+'_b',
                             initializer=tf.zeros(
                                 shape=self.rnn_config['shape']['att_linear_weight'][1],
                                 dtype='float32'),
                             dtype='float32')
        att_factor=tf.reshape(att_factor,shape=(-1,self.rnn_config['shape']['att_linear_weight'][0]))
        att_factor=tf.tanh(tf.matmul(att_factor,weight)+bias)
        graph.add_to_collection('att_factor_h', att_factor)
        weight= tf.get_variable(name=self.rnn_config['name']['attention']+'_w_sm',
                                initializer=tf.random_uniform(
                                    shape=self.rnn_config['shape']['att_softmax_weight'],
                                    dtype='float32'),
                                dtype='float32')
        graph.add_to_collection('reg',tf.contrib.layers.l2_regularizer(self.rnn_config['lamda'])(weight))
        bias=tf.get_variable(name=self.rnn_config['name']['attention']+'_b_sm',
                             initializer=tf.zeros(
                                 shape=self.rnn_config['shape']['att_softmax_weight'][1],
                                 dtype='float32'),
                             dtype='float32')
        att_factor=tf.matmul(att_factor,weight)+bias
        att_factor=tf.reshape(att_factor,shape=(self.rnn_config['sentence_num'],-1,self.rnn_config['sentence_words_num']))
        attention=tf.nn.softmax(att_factor)
        attention=tf.expand_dims(attention,axis=3)
        attention=tf.tile(attention,[1,1,1,self.rnn_config['word_dim']])
        graph.add_to_collection('attention',attention)
        return attention



    def sentence_rnn(self,x,graph):
        """

        :param x:
        :param graph:
        :return: [[[s11],[s12],...,[s1k]],...,[[sn1],[sn2],...,[snk]]]
        """
        weight=tf.get_variable(name=self.rnn_config['name']['rnn']+'_w',initializer=tf.random_uniform(shape=self.rnn_config['shape']['rnn']),dtype='float32')
        graph.add_to_collection('reg',tf.contrib.layers.l2_regularizer(self.rnn_config['lamda'])(weight))
        bias=tf.get_variable(name=self.rnn_config['name']['rnn']+'_b',initializer=tf.ones(shape=self.rnn_config['shape']['rnn'][1],dtype='float32'),dtype='float32')

        # [w1,w2,...] ==> s1
        # [[[w1,w2,...,wm],[w1,w2,..,wm],...],...]
        # ==>[[s11,s12,...s1k],...,[sn1,...,snk]]
        # snk is nth sentence of review k;
        collection=[]
        for count in range(self.rnn_config['sentence_num']):
            # with tf.variable_scope('sentence_lstm'+str(count)):
            n_steps=self.rnn_config['n_steps']
            cell_size=self.rnn_config['sentence_cell_size']
            X=tf.reshape(x[count],shape=(-1,self.rnn_config['word_dim']))
            Xt=tf.reshape(tf.matmul(X,weight)+bias,shape=(-1,n_steps,cell_size))
            if count == 0:
                reuse=None
            else:
                reuse=True
            cell=tf.nn.rnn_cell.BasicLSTMCell(cell_size,reuse=reuse)
            init_state=cell.zero_state(self.rnn_config['batch_size'],dtype='float32')
            outputs, _ = tf.nn.dynamic_rnn(cell,inputs=Xt,initial_state=init_state)
            # print(cell.name,' ',cell.scope_name)
            outputs=tf.transpose(outputs,[1,0,2])
            collection.append(tf.expand_dims(outputs[-1],1))
        graph.add_to_collection('sentence_rnn_output',collection)

        return collection

    def sentence_concat_with_aspect(self, A, collection, graph):
        # aspect is ordered like sentences: [[[s11],[s12],...,[s1k]],...,[[sn1],[sn2],...,[snk]]] <--> [[[a1],[a2],...[ak]], ..., [[a1],[a2],...,[ak]]]
        # [[s11],[s12],...,[s1k]],[[a1],[a2],...,[ak]] ==> [[s1,a2],...,[sk,ak]]
        # collection=[[[s1,a2],...,[sk,ak]], ...]
        A=self.aspect_transform(A,graph)
        new_collection=[]
        for i in range(len(collection)):
            sent=collection[i]
            aspect=A[i]
            # [s1,...,sk]+[a1,...,ak] ==> [[s1,a2],...,[sk,ak]]
            new_collection.append(tf.concat([sent,aspect],axis=2))
        graph.add_to_collection('sentence_concat_aspect',new_collection)
        return new_collection

    def hidden_layer(self,h,graph):
        hidden_layer_dim=self.rnn_config['hidden_layer_dim']
        if len(hidden_layer_dim) == 1:
            return h
        dim_pairs=[]
        for in_dim_pos in range(len(hidden_layer_dim)-1):
            out_dim_pos=in_dim_pos+1
            dim_pairs.append((hidden_layer_dim[in_dim_pos],hidden_layer_dim[out_dim_pos]))

        pre_dim=self.rnn_config['sentence_cell_size']+self.rnn_config['aspect_rnn_cell_size'] # need to add size of aspect
        for dim_pair in dim_pairs:
            weight=tf.get_variable(name='hidden',initializer=tf.random_uniform(shape=dim_pair,dtype='float32'),dtype='float32')
            bias=tf.get_variable(name='hidden',initializer=tf.ones(shape=dim_pair[1],dtype='float32'),dtype='float32')
            H=tf.reshape(h,shape=(-1,pre_dim))
            H=tf.matmul(H,weight)+bias
            H=tf.reshape(H,shape=(-1,self.rnn_config['batch_size'],dim_pair[1]))
            pre_dim=dim_pair[1]
        graph.add_to_collection('hidden_outputs',H)
        return H



    def sentence_concat_to_doc(self,collection,graph):
        Ds=tf.concat(collection,axis=1)
        graph.add_to_collection('Ds',Ds)
        return Ds

    def softmax(self,Ds,A,graph):
        Ds = tf.reshape(Ds,shape=[-1,self.rnn_config['sentence_cell_size']])
        A = tf.expand_dims(A,axis=1)
        A = tf.tile(A, multiples=[1,self.rnn_config['sentence_num'],1])
        A = tf.reshape(A,shape=(-1,self.rnn_config['aspect_rnn_cell_size']))
        weight=tf.get_variable(name=self.rnn_config['name']['softmax'] + '_w',
                                                            initializer=tf.random_uniform(
                                                                shape=(self.rnn_config['sentence_cell_size'],self.rnn_config['aspect_rnn_cell_size'])),
                                                            dtype='float32')
        graph.add_to_collection('reg',tf.contrib.layers.l2_regularizer(self.rnn_config['lamda'])(weight))
        bias=tf.get_variable(name=self.rnn_config['name']['softmax']+'_b',
                                                         initializer=tf.ones(
                                                             shape=(1,)),
                                                         dtype='float32')
        temp=tf.matmul(Ds,weight)
        Ds=tf.nn.tanh(tf.reduce_sum(tf.multiply(temp,A),axis=1)+bias)
        Ds=tf.reshape(Ds,shape=(-1,self.rnn_config['sentence_num']))
        softmax_result = tf.nn.softmax(Ds)
        graph.add_to_collection('softmax_result',softmax_result)
        return softmax_result

    def sigmoid(self,Ds,graph):
        Ds=tf.reshape(Ds,shape=(-1,self.rnn_config['sentence_cell_size']+self.rnn_config['aspect_rnn_cell_size']))
        weight=tf.get_variable(name=self.rnn_config['name']['sigmoid']+'_w',
                                                           initializer=tf.random_uniform(
                                                               shape=(self.rnn_config['sentence_cell_size']+self.rnn_config['aspect_rnn_cell_size'],1)),
                                                           dtype='float32')
        graph.add_to_collection('reg',tf.contrib.layers.l2_regularizer(self.rnn_config['lamda'])(weight))
        bias=tf.get_variable(name=self.rnn_config['name']['sigmoid']+'_b',
                                                         initializer=tf.ones(
                                                             shape=(1,)),
                                                         dtype='float32')
        Ds=tf.nn.tanh(tf.matmul(Ds,weight)+bias)
        Ds=tf.reshape(Ds,shape=(-1,self.rnn_config['sentence_num']))
        sigmoid_result = tf.nn.sigmoid(Ds)

        graph.add_to_collection('sigmoid_result',sigmoid_result)
        return sigmoid_result

    def cross_entropy(self,labels,logits,graph):
        J = tf.reduce_mean(tf.add(tf.multiply(labels, -tf.log(logits)), tf.multiply(tf.subtract(1.0, labels), -tf.log(tf.subtract(1.0, logits))))) \
            + tf.reduce_mean(tf.get_collection('reg'))
        graph.add_to_collection('loss',J)
        return J

    def optimizer(self,loss,graph):
        train_step=tf.train.AdamOptimizer(self.rnn_config['lr']).minimize(loss)
        graph.add_to_collection('train_step',train_step)
        return train_step

    def accuracy(self,gt,pre,graph):
        condition=tf.less(tf.zeros_like(pre)+0.5,pre)
        prediction=tf.where(condition,tf.ones_like(pre),tf.zeros_like(pre))
        condition=tf.equal(gt,prediction)
        accuracy=tf.reduce_mean(tf.where(condition,tf.ones_like(condition,dtype='float32'),tf.zeros_like(condition,dtype='float32')))
        graph.add_to_collection('accuracy',accuracy)
        return accuracy


    def classifier(self):
        graph=tf.Graph()
        with graph.as_default():
            #with tf.variable_scope
            # A is average of embeddings
            print('aspect')
            A = self.aspect(graph=graph)
            print('aspect_rnn')
            with tf.variable_scope('aspect_lstm'):
                A = self.aspect_rnn(A, graph=graph)
            print('X')
            X=self.input(graph=graph)

            # attention
            print('attention')
            attention=self.attention(x=X,a=A,graph=graph)
            # add attention to words
            print('attention*X')
            X=attention*X

            print('sentence rnn')
            # lstm for sentence
            with tf.variable_scope('sentence_lstm'):
                collection=self.sentence_rnn(x=X,graph=graph)
            # ======= =======
            # P(sentence|aspect)
            # ======= =======
            # [[[s11],...,[s1k]],...,[[sm1],...,[smk]]] ==> [[s11,s21,...,sm1],...,[s1k,s2k,...,smk]]
            print('sentences concatenate to document')
            Ds = self.sentence_concat_to_doc(collection=collection, graph=graph)
            # softmax to calculate P(s|aspect)P(s|D)
            print('softmax')
            softmax_outputs = self.softmax(Ds=Ds, A=A, graph=graph)


            # ======= =======
            # P(Ys|sentence,aspect)
            # ======= =======
            # concatenate aspect to each sentence
            print('sentence concatenate with aspect')
            collection = self.sentence_concat_with_aspect(A=A,collection=collection,graph=graph)
            # concatenate sentences to documents
            # [[[s11],...,[s1k]],...,[[sm1],...,[smk]]] ==> [[s11,s21,...,sm1],...,[s1k,s2k,...,smk]]
            # sij is the ith sentence in document j
            print('[sentence,aspect]s concatenate to documents')
            Ds=self.sentence_concat_to_doc(collection=collection,graph=graph)
            # sigmoid to calcualte P(Ys|s,aspect)
            print('sigmoid')
            sigmoid_outputs=self.sigmoid(Ds=Ds,graph=graph)

            # ======= =======
            # P(Ys|D,aspect)
            # ======= =======
            print('prediction')
            y=tf.reduce_sum(tf.multiply(softmax_outputs,sigmoid_outputs),axis=1)
            # ======= =======
            # training
            # ======= =======
            print('ground truth')
            y_=self.ground_truth(graph=graph)
            print('loss')
            loss=self.cross_entropy(labels=y_,logits=y,graph=graph)
            print('train steps')
            train_step=self.optimizer(loss=loss,graph=graph)
            print('accuracy')
            accuracy=self.accuracy(gt=y_,pre=y,graph=graph)
            saver=tf.train.Saver()
        return graph,saver

    def train(self):
        graph,saver=self.classifier()
        with graph.device('/gpu:0'):
            with graph.as_default():
                x=graph.get_collection('input')
                y_=graph.get_collection('ground_truth')
                train_step=graph.get_collection('train_step')
                accuracy=graph.get_collection('accuracy')
            with tf.Session(graph=graph,config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in self.rnn_config['epoch']:
                    sess.run(train_step,feed_dict={})
                    if epoch %20 ==0 and epoch!=0:
                        precision=sess.run(accuracy,feed_dict={})
                        with open(self.rnn_config['report'],'a+') as f:
                            f.write(str(precision)+'\n')
                            f.flush()
                saver.save(sess,self.rnn_config['sl_path'])