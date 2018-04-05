import sys
sys.path.append('/home/liu121/sentiment_coarse_model')

from sentiment.util.data_process.atr_data_generator import DataGenerator

class AttributeCount:
    def __init__(self,data_config):
        self.data_config = data_config
        dg = DataGenerator(data_config)
        self.labels_dic = dg.aspect_dic
        print(self.labels_dic)


if __name__ == "__main__":
    data_config = {
        'train_source_file_path': '/datastore/liu121/senti_data/semeval2016/absa_resturant_train.csv',
        'train_data_file_path': '/datastore/liu121/senti_data/semeval2016/restaurant_attribute_data_train.pkl',
        'test_source_file_path': '/datastore/liu121/senti_data/semeval2016/absa_resturant_test.csv',
        'test_data_file_path': '/datastore/liu121/senti_data/semeval2016/restaurant_attribute_data_test.pkl',
    }
    ac = AttributeCount(data_config)
