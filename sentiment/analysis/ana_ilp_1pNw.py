import os
import sys
if os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.transfer_nn.ilp_1pNw.classifier import Classifier

class Analysis:
    def __init__(self, coarse_nn_config, fine_nn_config, coarse_data_generator, fine_data_generator):
        cl = Classifier(coarse_nn_config, fine_nn_config, coarse_data_generator, fine_data_generator)


    def