import tensorflow as tf
import numpy as np

class Metrics:
    def __init__(self,nn_config):
        self.nn_config = nn_config

    def sentiment_f1(self, label, pred, graph):
        """

        :param Y_att: shape = (batch size, attributes number+1, 3)
        :param pred: shape = (batch size, attributes number+1, 3)
        :param graph: 
        :return: 
        """
        label = tf.reshape(label,[self.nn_config['batch_size'],-1])
        pred  = tf.reshape(pred, [self.nn_config['batch_size'], -1])

        # TP.shape = (batch size, attributes number+1 * 3)
        TP = tf.cast(tf.count_nonzero(pred * label, axis=0,keep_dims=True), tf.float32)
        # TP.shape = (batch size, attributes number+1)
        TP = tf.reduce_sum(tf.reshape(TP,shape=[self.nn_config['batch_size'],self.nn_config['attributes_num']+1,3]),axis=2)


        # FP.shape = (batch size, attributes number+1 * 3)
        FP = tf.cast(tf.count_nonzero(pred * (label - 1), axis=0), tf.float32)
        # FP.shape = (batch size, attributes number+1)
        FP = tf.reduce_sum(tf.reshape(FP, shape=[self.nn_config['batch_size'], self.nn_config['attributes_num'] + 1, 3]), axis=2)

        # FN.shape = (batch size, attributes number+1 * 3)
        FN = tf.cast(tf.count_nonzero((pred - 1) * label, axis=0), tf.float32)
        # FN.shape = (batch size, attributes number+1)
        FN = tf.reduce_sum(tf.reshape(FN, shape=[self.nn_config['batch_size'], self.nn_config['attributes_num'] + 1, 3]), axis=2)
        graph.add_to_collection('TP', TP)
        graph.add_to_collection('FP', FP)
        graph.add_to_collection('FN', FN)

        precision = tf.divide(TP, tf.add(TP + FP, 0.001))
        graph.add_to_collection('precision', precision)

        recall = tf.divide(TP, tf.add(TP + FN, 0.001))
        graph.add_to_collection('recall', recall)
        f1 = tf.divide(2 * precision * recall, tf.add(precision + recall, 0.001))
        graph.add_to_collection('f1', f1)

        # graph.add_to_collection('accuracy',accuracy)

        return f1

    def precision(self, TP, FP, flag):
        assert flag == 'macro' or flag == 'micro', 'Please enter right flag...'
        if flag == 'macro':
            tmp = np.nonzero((np.sum(TP, axis=0) + np.sum(FP, axis=0) == 0))
            res = np.sum(TP, axis=0, dtype='float32') / (np.sum(TP, axis=0, dtype='float32') + np.sum(FP, axis=0, dtype='float32'))
            res[tmp] = 1
            return res
        else:
            return np.sum(TP) / (np.sum(TP) + np.sum(FP) + 1e-10)

    def recall(self, TP, FN, flag):
        assert flag == 'macro' or flag == 'micro', 'Please enter right flag...'
        if flag == 'macro':
            tmp = np.nonzero((np.sum(TP, axis=0) + np.sum(FN, axis=0) == 0))
            res = np.sum(TP, axis=0, dtype='float32') / (
            np.sum(TP, axis=0, dtype='float32') + np.sum(FN, axis=0, dtype='float32'))
            res[tmp] = 1
            return res
        else:
            return np.sum(TP) / (np.sum(TP) + np.sum(FN) + 1e-10)

    def f1_score(self, precision, recall, flag):
        assert flag == 'macro' or flag == 'micro', 'Please enter right flag...'
        if flag == 'macro':
            tmp = np.nonzero((precision + recall) == 0)
            res = 2 * precision * recall / (precision + recall + 1e-10)
            res[tmp] = 0
            return res
        else:
            return 2 * precision * recall / (precision + recall + 1e-10)


    def accuracy(self, Y_att, pred, graph):
        """

        :param Y_att: shape = (batch size, attributes number)
        :param pred: shape = (batch size, attributes number)
        :param graph: 
        :return: 
        """
        # condition = tf.equal(Y_att, pred)
        # cmp = tf.reduce_sum(tf.where(condition,tf.zeros_like(Y_att,dtype='float32'),tf.ones_like(Y_att,dtype='float32')),axis=1)
        # condition = tf.equal(cmp,tf.zeros_like(cmp))
        # accuracy = tf.reduce_mean(tf.where(condition,tf.ones_like(cmp,dtype='float32'),tf.zeros_like(cmp,dtype='float32')))
        # accuracy = tf.reduce_mean(tf.where(condition, tf.ones_like(Y_att, dtype='float32'), tf.zeros_like(Y_att, dtype='float32')))
        # graph.add_to_collection('accuracy', accuracy)

        TP = tf.cast(tf.count_nonzero(pred * Y_att, axis=0), tf.float32)

        TN = tf.cast(tf.count_nonzero((pred - 1) * (Y_att - 1), axis=0), tf.float32)
        FP = tf.cast(tf.count_nonzero(pred * (Y_att - 1), axis=0), tf.float32)
        graph.add_to_collection('TP', TP)
        graph.add_to_collection('FP', FP)

        FN = tf.cast(tf.count_nonzero((pred - 1) * Y_att, axis=0), tf.float32)
        graph.add_to_collection('FN', FN)

        precision = tf.divide(TP, tf.add(TP + FP, 0.001))
        graph.add_to_collection('precision', precision)

        recall = tf.divide(TP, tf.add(TP + FN, 0.001))
        graph.add_to_collection('recall', recall)
        f1 = tf.divide(2 * precision * recall, tf.add(precision + recall, 0.001))
        graph.add_to_collection('f1', f1)

        return f1

    # nw
    # def precision(self,TP,FP,flag):
    #     assert flag=='macro' or flag=='micro','Please enter right flag...'
    #     if flag == 'macro':
    #         tmp = np.nonzero((np.sum(TP,axis=0) + np.sum(FP,axis=0) == 0))
    #         res = np.sum(TP,axis=0,dtype='float32') / ( np.sum(TP,axis=0,dtype='float32') + np.sum(FP,axis=0,dtype='float32') )
    #         res[tmp] = 1
    #         return res
    #     else:
    #         return np.sum(TP) / ( np.sum(TP) + np.sum(FP) )
    #
    # def recall(self,TP,FN,flag):
    #     assert flag=='macro' or flag=='micro','Please enter right flag...'
    #     if flag == 'macro':
    #         tmp = np.nonzero((np.sum(TP, axis=0) + np.sum(FN, axis=0) == 0))
    #         res = np.sum(TP, axis=0 ,dtype='float32') / (np.sum(TP, axis=0,dtype='float32') + np.sum(FN, axis=0,dtype='float32'))
    #         res[tmp] = 1
    #         return res
    #     else:
    #         return np.sum(TP) / ( np.sum(TP) + np.sum(FN) )
    #
    #
    # def f1_score(self,precision,recall,flag):
    #     assert flag=='macro' or flag=='micro','Please enter right flag...'
    #     if flag == 'macro':
    #         tmp = np.nonzero((precision + recall) == 0)
    #         res = 2 * precision * recall / ( precision + recall + 1e-10)
    #         res[tmp] = 0
    #         return res
    #     else:
    #         return 2 * precision * recall / ( precision + recall + 1e-10)

    # 1pNW
    # def accuracy(self, Y_att, pred, graph):
    #     """
    #
    #     :param Y_att: shape = (batch size, attributes number)
    #     :param pred: shape = (batch size, attributes number)
    #     :param graph:
    #     :return:
    #     """
    #
    #
    #     TP = tf.cast(tf.count_nonzero(pred * Y_att,axis=0), tf.float32)
    #
    #     TN = tf.cast(tf.count_nonzero((pred - 1) * (Y_att - 1),axis=0), tf.float32)
    #     FP = tf.cast(tf.count_nonzero(pred * (Y_att - 1),axis=0), tf.float32)
    #     graph.add_to_collection('TP', TP)
    #     graph.add_to_collection('FP', FP)
    #
    #     FN = tf.cast(tf.count_nonzero((pred - 1) * Y_att,axis=0), tf.float32)
    #     graph.add_to_collection('FN', FN)
    #
    #     return TP,TN,FP,FN
    #
    # def precision(self,TP,FP,flag):
    #     assert flag=='macro' or flag=='micro','Please enter right flag...'
    #     if flag == 'macro':
    #         tmp = np.nonzero((np.sum(TP,axis=0) + np.sum(FP,axis=0) == 0))
    #         res = np.sum(TP,axis=0,dtype='float32') / ( np.sum(TP,axis=0,dtype='float32') + np.sum(FP,axis=0,dtype='float32') )
    #         res[tmp] = 1
    #         return res
    #     else:
    #         return np.sum(TP) / ( np.sum(TP) + np.sum(FP) )
    #
    # def recall(self,TP,FN,flag):
    #     assert flag=='macro' or flag=='micro','Please enter right flag...'
    #     if flag == 'macro':
    #         tmp = np.nonzero((np.sum(TP, axis=0) + np.sum(FN, axis=0) == 0))
    #         res = np.sum(TP, axis=0 ,dtype='float32') / (np.sum(TP, axis=0,dtype='float32') + np.sum(FN, axis=0,dtype='float32'))
    #         res[tmp] = 1
    #         return res
    #     else:
    #         return np.sum(TP) / ( np.sum(TP) + np.sum(FN) )
    #
    #
    # def f1_score(self,precision,recall,flag):
    #     assert flag=='macro' or flag=='micro','Please enter right flag...'
    #     if flag == 'macro':
    #         tmp = np.nonzero((precision + recall) == 0)
    #         res = 2 * precision * recall / ( precision + recall + 1e-10)
    #         res[tmp] = 0
    #         return res
    #     else:
    #         return 2 * precision * recall / ( precision + recall + 1e-10)