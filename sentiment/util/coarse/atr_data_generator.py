import numpy as np
class DataGenerator:
    def __init__(self,nn_config):
        self.nn_config = nn_config
        # data = [[review, is_padding, attribute labels],...]; its type is list
        self.data = np.array([1,2,3])
        np.random.shuffle(self.data)

    def table_generator(self):
        return np.random.uniform(size=(3000,200)).astype('float32')

    def data_generator(self,mode,**kwargs):
        """
        
        :param mode: only 'train' or 'test'
        :return: 
        """
        if mode == 'train':
            batch_num=kwargs['batch_num']
            data_temp=self.data[:-self.nn_config['test_data_size']]
            train_data_size = len(data_temp)
            start = batch_num * self.nn_config['batch_size'] % train_data_size
            end = (batch_num * self.nn_config['batch_size'] + self.nn_config['batch_size']) % train_data_size
            if start < end:
                batch = data_temp[start:end]
            elif start >= end:
                batch = data_temp[start:]
                batch.extend(data_temp[0:end])
        elif mode =='test':
            data_temp=self.data[-self.nn_config['test_data_size']:]
            batch = data_temp

        return batch[:,0],batch[:,1],batch[:,2]