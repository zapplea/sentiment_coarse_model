import pickle
import numpy as np

def train(infile,outfile):
    with open(infile,'rb') as f:
        attribute_dic, word_dic, attr_labels, senti_labels, sentence, word_embed = pickle.load(f)
        non_attr = np.zeros((attr_labels.shape[0],1),dtype='float32')
        non_attr_senti = np.tile(non_attr,reps=[1,3])
        non_attr_senti = np.expand_dims(non_attr_senti,axis=1)
        senti_labels = np.concatenate([senti_labels,non_attr_senti],axis=1)

    with open(outfile,'wb') as f:
        pickle.dump((attribute_dic, word_dic, attr_labels[:20], senti_labels[:20], sentence[:20], word_embed),f)

def test(infile,outfile):
    with open(infile,'rb') as f:
        attr_labels, senti_labels, sentence = pickle.load(f)
        non_attr = np.zeros((attr_labels.shape[0],1),dtype='float32')
        non_attr_senti = np.tile(non_attr,reps=[1,3])
        non_attr_senti = np.expand_dims(non_attr_senti,axis=1)
        senti_labels = np.concatenate([senti_labels,non_attr_senti],axis=1)

    with open(outfile,'wb') as f:
        pickle.dump((attr_labels[:20], senti_labels[:20], sentence[:20]),f)

if __name__=='__main__':
    path = {'coarse_train_in':'/datastore/liu121/sentidata2/data/aic2018/coarse_data_backup/train_coarse.pkl',
            'coarse_train_out':'/datastore/liu121/sentidata2/data/aic2018/coarse_data/train_coarse.pkl',
            'coarse_test_in':'/datastore/liu121/sentidata2/data/aic2018/coarse_data_backup/dev_coarse.pkl',
            'coarse_test_out':'/datastore/liu121/sentidata2/data/aic2018/coarse_data/dev_coarse.pkl',
            'fine_train_in':'/datastore/liu121/sentidata2/data/aic2018/fine_data_backup/train_fine.pkl',
            'fine_train_out':'/datastore/liu121/sentidata2/data/aic2018/fine_data/train_fine.pkl',
            'fine_test_in':'/datastore/liu121/sentidata2/data/aic2018/fine_data_backup/dev_fine.pkl',
            'fine_test_out':'/datastore/liu121/sentidata2/data/aic2018/fine_data/dev_fine.pkl',
           }
    train(path['coarse_train_in'],path['coarse_train_out'])
    test(path['coarse_test_in'],path['coarse_test_out'])