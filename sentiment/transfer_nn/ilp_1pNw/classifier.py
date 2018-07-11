import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')
from sentiment.coarse_nn.coarse_atr_classifier_1pNw_bilstm.classifier import Classifier as coarse_Classifier
from sentiment.sep_nn.fine_atr_classifier_1pNw_bilstm.classifier import Classifier as fine_Classifier
from sentiment.transfer_nn.transfer.transfer_ilp import Transfer
from sentiment.functions.train.trans_atr_train_bilstm import TransferTrain

class Classifier:
    def __init__(self, coarse_nn_config, fine_nn_config, coarse_data_generator, fine_data_generator):
        self.coarse_nn_config = coarse_nn_config
        self.coarse_dg = coarse_data_generator

        self.fine_nn_config = fine_nn_config
        self.fine_dg = fine_data_generator

        self.trans=Transfer(coarse_nn_config,self.coarse_dg)
        self.tra=TransferTrain(fine_nn_config,self.fine_dg)

    def coarse_classifier(self):
        coarse_cl = coarse_Classifier(self.coarse_nn_config, self.coarse_dg)
        return coarse_cl

    def transfer(self,coarse_cl):
        init_data = self.trans.transfer(coarse_cl,self.fine_dg)
        return init_data


    def fine_classifier(self):
        fine_cl = fine_Classifier(self.fine_nn_config, self.fine_dg)
        return fine_cl

    def classifier(self):
        coarse_cl = self.coarse_classifier()
        init_data = self.transfer(coarse_cl)
        fine_cl = self.fine_classifier()
        return fine_cl,init_data

    def train(self):
        """
        train fine grained model with 
        :return: 
        """
        fine_cl,init_data=self.classifier()
        self.tra.train(fine_cl,init_data=init_data)