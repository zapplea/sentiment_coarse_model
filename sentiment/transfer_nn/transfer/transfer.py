import tensorflow as tf
import numpy as np

class Transfer:
    def __init__(self,fine_nn_config,coarse_nn_config, coarse_data_generator):
        self.fine_nn_config=fine_nn_config
        self.coarse_nn_config = coarse_nn_config
        self.coarse_data_generator = coarse_data_generator

    def attribute_vec(self, graph):
        """

        :param graph: 
        :return: shape = (number of attributes+1, attributes dim)
        """
        # A is matrix of attribute vector
        initializer_A= tf.placeholder(shape=(self.fine_nn_config['attributes_num'],self.fine_nn_config['attribute_dim']),dtype='float32')
        graph.add_to_collection('initializer_A',initializer_A)
        A = tf.Variable(initial_value= initializer_A)
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.fine_nn_config['reg_rate'])(A))
        graph.add_to_collection('A_vec', A)
        initializer_O = tf.placeholder(shape=(1, self.fine_nn_config['attribute_dim']),dtype='float32')
        graph.add_to_collection('initializer_O',initializer_O)
        o = tf.Variable(initial_value=initializer_O)
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.fine_nn_config['reg_rate'])(o))
        graph.add_to_collection('o_vec', o)
        return A, o

    def attribute_mat(self, graph):
        """

        :param graph: 
        :return: shape = (attributes number+1, attribute mat size, attribute dim)
        """
        initializer_A = tf.placeholder(shape=(self.fine_nn_config['attributes_num'],
                                              self.fine_nn_config['attribute_mat_size'],
                                              self.fine_nn_config['attribute_dim']),
                                       dtype='float32')
        graph.add_to_collection('initializer_A', initializer_A)
        A_mat = tf.Variable(initial_value=initializer_A)
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.fine_nn_config['reg_rate'])(A_mat))
        graph.add_to_collection('A_mat', A_mat)

        initializer_O=tf.placeholder(shape=(1,
                                            self.fine_nn_config['attribute_mat_size'],
                                            self.fine_nn_config['attribute_dim']),
                                     dtype='float32')
        graph.add_to_collection('initializer_O', initializer_O)
        o_mat = tf.Variable(initial_value=initializer_O)
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.fine_nn_config['reg_rate'])(o_mat))
        graph.add_to_collection('o_mat', o_mat)
        return A_mat,o_mat

    def softmax(self,score):
        """
        
        :param score: (sentences number, attributes num)
        :return: 
        """
        avg_score = tf.reduce_mean(score,axis=0)
        index = tf.argmax(tf.nn.softmax(avg_score,axis=0))
        return index

    def transfer(self,coarse_model,fine_dg):
        graph,saver = coarse_model.classifier()
        score = graph.get_collection('score')[0]
        X = graph.get_collection('X')[0]
        keep_prob_lstm = graph.get_collection('keep_prob_lstm')[0]
        index = self.softmax(score)

        # X_data.shape = (fine grained attributes number, number of sentences,1,words num)
        X_data_list = self.coarse_data_generator.fine_sentences(fine_dg.train_attribute_ground_truth,fine_dg.train_sentence_ground_truth)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        index_list = []
        initializer_A = []
        with tf.Session(graph=graph,config=config) as sess:
            model_file = tf.train.latest_checkpoint(self.coarse_nn_config['sr_path'])
            saver.restore(sess,model_file)
            for X_data in X_data_list:
                index_data,score_data = sess.run([index,score],feed_dict={X:X_data,keep_prob_lstm:1.0})
                print(index_data,np.mean(score_data,axis=0))
                index_list.append(index_data)
            if self.coarse_nn_config['is_mat']:
                A = graph.get_collection('A_mat')[0]
                O = graph.get_collection('o_mat')[0]
            else:
                A = graph.get_collection('A_vec')[0]
                O = graph.get_collection('o_vec')[0]
            A_data,initializer_O = sess.run([A,O])
            for index_data in index_list:
                initializer_A.append(A_data[index_data.astype('int32')])
        return initializer_A,initializer_O