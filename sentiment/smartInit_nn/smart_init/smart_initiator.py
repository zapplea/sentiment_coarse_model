import tensorflow as tf

class SmartInitiator:
    def __init__(self, nn_config):
        self.nn_config = nn_config


    def smart_initiater(self,graph):
        """
        :param attributes: ndarray, shape=(attribute numbers ,2, attribute dim)
        :return: 
        """
        # TODO: need to be careful to one attribute: STYLE_OPTIONS, should split them to two words and mean.
        # random_mat.shape = (attributes number, attribute mat size-2, attribute dim)
        mentions_mat = tf.placeholder(shape=(self.nn_config['attributes_num'],
                                             2,
                                             self.nn_config['attribute_dim']),
                                      dtype='float32')
        random_mat = tf.random_normal(shape=(self.nn_config['attributes_num'],
                                             self.nn_config['attribute_mat_size'] - 2,
                                             self.nn_config['attribute_dim']),
                                      dtype='float32')
        attributes_mat = tf.nn.l2_normalize(tf.concat([mentions_mat, random_mat], axis=1),axis=2)
        graph.add_to_collection('smartInit',attributes_mat)
        return attributes_mat

    def attribute_mat(self, smartInit, graph):
        """

        :param graph: 
        :return: shape = (attributes number+1, attribute mat size, attribute dim)
        """
        A_mat = tf.get_variable(name='A_mat', initializer=smartInit)
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(A_mat))
        graph.add_to_collection('A_mat', A_mat)
        o_mat = tf.get_variable(name='other_vec',
                                initializer=tf.random_uniform(shape=(1,
                                                                     self.nn_config['attribute_mat_size'],
                                                                     self.nn_config['attribute_dim']),
                                                              dtype='float32'))
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(o_mat))
        graph.add_to_collection('o_mat', o_mat)
        return A_mat,o_mat