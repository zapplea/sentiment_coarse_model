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

