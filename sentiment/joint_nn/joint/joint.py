import tensorflow as tf
class Joint:
    def __init__(self,nn_config):
        self.nn_config=nn_config

    def expand_attr_labels(self, labels):
        """

        :param graph: 
        :return: shape = (batch size, attributes number+1)
        """

        Y_att=labels
        # TODO: add non-attribute
        batch_size = tf.shape(Y_att)[0]
        non_attr = tf.zeros((batch_size,1),dtype='float32')
        condition = tf.equal(tf.reduce_sum(Y_att,axis=1,keepdims=True),non_attr)
        non_attr = tf.where(condition,tf.ones_like(non_attr),non_attr)
        Y_att = tf.concat([Y_att,non_attr],axis=1)
        return Y_att

    def joint_optimizer(self,senti_loss,attr_loss,graph):
        """
        
        :param senti_loss: 
        :param attr_loss: 
        :return: 
        """
        loss = senti_loss+attr_loss
        opt = tf.train.AdamOptimizer(self.nn_config['joint_lr']).minimize(loss)
        graph.add_to_collection('joint_opt', opt)
        return opt