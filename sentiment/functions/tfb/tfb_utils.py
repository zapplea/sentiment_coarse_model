import tensorflow as tf

class Tfb:
    def __init__(self,fine_nn_config):
        self.fine_nn_config = fine_nn_config

    def scalar(self):
        micro_f1 = tf.get_variable(name='micro_f1',initializer=tf.zeros(shape=(),dtype='float32'))
        tf.summary.scalar('micro_f1',micro_f1)
        micro_pre = tf.get_variable(name='micro_pre',initializer=tf.zeros(shape=(),dtype='float32'))
        tf.summary.scalar('micro_pre',micro_pre)
        micro_rec = tf.get_variable(name='micro_rec', initializer=tf.zeros(shape=(), dtype='float32'))
        tf.summary.scalar('micro_rec', micro_rec)

        macro_f1 = tf.get_variable(name='macro_f1', initializer=tf.zeros(shape=(), dtype='float32'))
        tf.summary.scalar('macro_f1', macro_f1)
        macro_pre = tf.get_variable(name='macro_pre', initializer=tf.zeros(shape=(), dtype='float32'))
        tf.summary.scalar('macro_pre', macro_pre)
        macro_rec = tf.get_variable(name='macro_rec', initializer=tf.zeros(shape=(), dtype='float32'))
        tf.summary.scalar('macro_rec', macro_rec)

        tfb_loss = tf.get_variable(name='tfb_loss',initializer=tf.zeros(shape=(),dtype='float32'))

        return micro_f1,micro_pre,micro_rec,macro_f1,macro_pre,macro_rec, tfb_loss

    def mat_emb(self,fine_atr_mat,coarse_atr_mat):
        """
        
        :param fine_atr_mat: shape = (fine attributes num, fine mat size, attribute dim)
        :param coarse_atr_mat: shape = (coarse attributes num, coarse mat size, attribute dim)
        :return: 
        """
        coarse_atr_mat = tf.reduce_mean(coarse_atr_mat,axis=1)
        coarse_embedding = tf.Variable(
            tf.zeros([self.fine_nn_config['coarse_attributes_num'], self.fine_nn_config['attribute_dim']]), dtype='float32',
            name='coarse_atr_mat_embedding')
        coarse_assignment = coarse_embedding.assign(coarse_atr_mat)

        fine_atr_mat = tf.reduce_mean(fine_atr_mat,axis=1)
        fine_embedding = tf.Variable(tf.zeros([self.fine_nn_config['attributes_num'],self.fine_nn_config['attribute_dim']]),
                                     dtype='float32',name='fine_atr_mat_embedding')
        fine_assignment = fine_embedding.assign(fine_atr_mat)

        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()

        coarse_embedding_config = config.embeddings.add()
        coarse_embedding_config.tensor_name = coarse_embedding.name
        coarse_embedding_config.metadata_path = self.fine_nn_config['tfb_coarse_atr_matPath']

        fine_embedding_config = config.embeddings.add()
        fine_embedding_config.tensor_name = fine_embedding.name
        fine_embedding_config.metadata_path = self.fine_nn_config['tfb_fine_atr_matPath']

        return config, fine_assignment,coarse_assignment
