import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.functions.attribute_function.metrics import Metrics

class JointTrain:
    def __init__(self,nn_config, data_generator):
        self.nn_config = nn_config
        self.dg = data_generator
        self.mt = Metrics(self.nn_config)

    def train(self,classifier):
        graph, saver = classifier