import os
import pickle

def load_train_data(data_config):
    if os.path.exists(data_config['train_data_file_path']) and os.path.getsize(
            data_config['train_data_file_path']) > 0:
        with open(data_config['train_data_file_path'], 'rb') as f:
            attribute_dic, word_dic, attr_labels, senti_labels, sentence, word_embed = pickle.load(f)
        print(word_dic.keys())
        exit()
        return attr_labels, senti_labels, sentence, attribute_dic, word_dic, word_embed

if __name__ == "__main__":
    data_config = {'train_data_file_path':'/datastore/liu121/sentidata2/data/aic2018/coarse_data/train_coarse.pkl'}
    load_train_data(data_config)