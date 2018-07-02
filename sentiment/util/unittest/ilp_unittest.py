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
        self.ilp_data=self.data_generator()
        self.ilp=AttributeIlp(self.ilp_data)

    def data_generator(self):
        data={}
        # score_pre.shape = (batch size, aspects num, words num)
        # attention.shape = (batch size, words number, aspects num, aspect mat size)
        for i in range(11):
            batch_size =2*(i+1)
            aspects_num = 5
            words_num = 10
            mat_size = 3
            score_pre_data = np.random.normal(size=(batch_size,aspects_num, words_num))
            attention_data = np.random.normal(size=(batch_size, words_num, aspects_num, mat_size))
            data[i]={'score_pre':score_pre_data,'attention':attention_data}
        return data

    def test_attributes_vec_index(self):
        index_collection = self.ilp.attributes_vec_index()

    def test_attributes_matrix(self):
        aspects_num = 5
        mat_size = 3
        attr_dim = 300

        index_collection = self.ilp.attributes_vec_index()
        matrix = np.random.normal(size=(aspects_num*mat_size,attr_dim))
        A_data= self.ilp.attributes_matrix(index_collection,matrix)

if __name__=="__main__":
    unittest.main()