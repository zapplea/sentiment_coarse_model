import os
import sys
if os.getlogin() == 'yibing':
    sys.path.append('/home/yibing/Documents/csiro/sentiment_coarse_model')
elif os.getlogin() == 'lujunyu':
    sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
elif os.getlogin() == 'liu121':
    sys.path.append('/home/liu121/sentiment_coarse_model')

import unittest

from sentiment.functions.ilp.ilp import AttributeIlp
import numpy as np

class TestAttrIlp(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(TestAttrIlp, self).__init__(*args, **kwargs)
        self.aspects_num = 5
        self.words_num = 10
        self.mat_size = 3
        self.ilp_data=self.data_generator()
        self.ilp=AttributeIlp(self.ilp_data,self.aspects_num,self.mat_size)

    def data_generator(self):
        data={}
        # score_pre.shape = (batch size, 1, aspects num*words num)
        # attention.shape = (batch size, words number, 1, aspects num*aspect mat size)
        for i in range(11):
            batch_size =2*(i+1)

            score_pre_data = np.random.normal(size=(batch_size,1, self.words_num))
            attention_data = np.random.normal(size=(batch_size, self.words_num, 1, self.aspects_num*self.mat_size))
            data[i]={'score_pre':score_pre_data,'attention':attention_data}
        return data

    def test_attributes_vec_index(self):
        index_collection = self.ilp.attributes_vec_index()

    def test_attributes_matrix(self):
        attr_dim = 200
        index_collection = self.ilp.attributes_vec_index()
        matrix = np.random.normal(size=(self.aspects_num*self.mat_size,attr_dim))
        A_data= self.ilp.attributes_matrix(index_collection,matrix)

if __name__=="__main__":
    unittest.main()