import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.functions.attribute_function.metrics import Metrics
import tensorflow as tf
import numpy as np

class SentiTrain:
    def __init__(self,nn_config, data_generator):
        self.nn_config = nn_config
        self.dg = data_generator

    def train(self,classifier):
        graph, saver = classifier()
        with graph.as_default():
            # input
            X = graph.get_collection('X')[0]
            # labels
            Y_att = graph.get_collection('Y_att')[0]
            Y_senti = graph.get_collection('Y_senti')[0]
            # lookup table
            table = graph.get_collection('table')[0]
            # train_step
            train_step = graph.get_collection('train_step')[0]
            # loss
            loss = graph.get_collection('senti_loss')[0]
            # accuracy
            accuracy = graph.get_collection('accuracy')[0]
            # attribute function
            init = tf.global_variables_initializer()
        with graph.device('/gpu:1'):
            table_data = self.dg.table_generator()
            with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                sess.run(init, feed_dict={table: table_data})
                for i in range(self.nn_config['epoch']):
                    sentences, Y_att_data, Y_senti_data = self.dg.data_generator(i)
                    sess.run(train_step, feed_dict={X: sentences, Y_att: Y_att_data, Y_senti: Y_senti_data})

                    if i % 5000 == 0 and i != 0:
                        sentences, Y_att_data, Y_senti_data = self.dg.data_generator('test')
                        valid_size = Y_att_data.shape[0]
                        p = 0
                        l = 0
                        count = 0
                        batch_size = self.nn_config['batch_size']
                        for i in range(valid_size // batch_size):
                            count += 1
                            p += sess.run(accuracy, feed_dict={X: sentences[i * batch_size:i * batch_size + batch_size],
                                                               Y_att: Y_att_data[
                                                                      i * batch_size:i * batch_size + batch_size],
                                                               Y_senti: Y_senti_data[
                                                                        i * batch_size:i * batch_size + batch_size]})
                            l += sess.run(loss, feed_dict={X: sentences[i * batch_size:i * batch_size + batch_size],
                                                           Y_att: Y_att_data[
                                                                  i * batch_size:i * batch_size + batch_size],
                                                           Y_senti: Y_senti_data[
                                                                    i * batch_size:i * batch_size + batch_size]})
                        p = p / count
                        l = l / count
