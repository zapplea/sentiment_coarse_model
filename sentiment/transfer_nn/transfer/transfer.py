import tensorflow as tf

class Transfer:
    def __init__(self,nn_config):
        self.nn_config=nn_config

    def attribute_vec(self, initializer_A,initializer_O, graph):
        """

        :param graph: 
        :return: shape = (number of attributes+1, attributes dim)
        """
        # A is matrix of attribute vector
        A = tf.Variable(initial_value= initializer_A)
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(A))
        graph.add_to_collection('A_vec', A)
        o = tf.Variable(initial_value=initializer_O)
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(o))
        graph.add_to_collection('o_vec', o)
        return A, o

    def attribute_mat(self,initializer_A,initializer_O, graph):
        """

        :param graph: 
        :return: shape = (attributes number+1, attribute mat size, attribute dim)
        """
        A_mat = tf.Variable(initial_value=initializer_A)
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(A_mat))
        graph.add_to_collection('A_mat', A_mat)
        o_mat = tf.Variable(initial_value=initializer_O)
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(o_mat))
        graph.add_to_collection('o_mat', o_mat)
        return A_mat,o_mat