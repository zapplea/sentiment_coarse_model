import pickle
import numpy as np
import argparse


def train(infile,outfile, top_k):
    print('train:')
    with open(infile,'rb') as f:
        attribute_dic, word_dic, attr_labels, senti_labels, sentence, word_embed = pickle.load(f)
        print('shape of sentence: ',sentence.shape)
        print('shape of attributes: ', attr_labels.shape)
        print('shape of senti labels: ', senti_labels.shape)

        non_attr = np.zeros((attr_labels.shape[0],1),dtype='float32')
        non_attr_senti = np.tile(non_attr,reps=[1,3])
        non_attr_senti = np.expand_dims(non_attr_senti,axis=1)
        senti_labels = np.concatenate([senti_labels,non_attr_senti],axis=1)
        attr_labels=attr_labels[:top_k]
        senti_labels=senti_labels[:top_k]
        sentence=sentence[:top_k]
        print('senti labels shape: ',senti_labels.shape)
        print('attr labels shape: ',attr_labels.shape)
        print('sentence shape: ',sentence.shape)
    with open(outfile,'wb') as f:
        pickle.dump((attribute_dic, word_dic, attr_labels, senti_labels, sentence, word_embed),f)
    print('train successful\n')

def test(infile,outfile, top_k):
    print('test:')
    with open(infile,'rb') as f:
        attr_labels, senti_labels, sentence = pickle.load(f)
        print('shape of sentence: ', sentence.shape)
        print('shape of attributes: ', attr_labels.shape)
        print('shape of senti labels: ', senti_labels.shape)

        non_attr = np.zeros((attr_labels.shape[0],1),dtype='float32')
        non_attr_senti = np.tile(non_attr,reps=[1,3])
        non_attr_senti = np.expand_dims(non_attr_senti,axis=1)
        senti_labels = np.concatenate([senti_labels,non_attr_senti],axis=1)
        attr_labels = attr_labels[:top_k]
        senti_labels = senti_labels[:top_k]
        sentence = sentence[:top_k]
        print('senti labels shape: %s'%str(senti_labels.shape))
        print('attr labels shape: %s'%str(attr_labels.shape))
        print('sentence shape: %s'%str(sentence.shape))
    with open(outfile,'wb') as f:
        pickle.dump((attr_labels, senti_labels, sentence),f)
    print('test success')

def fine_train(infile,outfile, top_k):
    print('train:')
    with open(infile,'rb') as f:
        attribute_dic, word_dic, attr_labels, senti_labels, sentence, word_embed = pickle.load(f)
        print('shape of sentence: ',sentence.shape)
        print('shape of attributes: ', attr_labels.shape)
        print('shape of senti labels: ', senti_labels.shape)
        senti_labels = senti_labels[:,:,:-1]
        non_attr = np.zeros((attr_labels.shape[0],1),dtype='float32')
        non_attr_senti = np.tile(non_attr,reps=[1,3])
        non_attr_senti = np.expand_dims(non_attr_senti,axis=1)
        senti_labels = np.concatenate([senti_labels,non_attr_senti],axis=1)
        attr_labels=attr_labels[:top_k]
        senti_labels=senti_labels[:top_k]
        sentence=sentence[:top_k]
        print('senti labels shape: ',senti_labels.shape)
        print('attr labels shape: ',attr_labels.shape)
        print('sentence shape: ',sentence.shape)
    with open(outfile,'wb') as f:
        pickle.dump((attribute_dic, word_dic, attr_labels, senti_labels, sentence, word_embed),f)
    print('train successful\n')

def fine_test(infile,outfile, top_k):
    print('test:')
    with open(infile,'rb') as f:
        attr_labels, senti_labels, sentence = pickle.load(f)
        print('shape of sentence: ', sentence.shape)
        print('shape of attributes: ', attr_labels.shape)
        print('shape of senti labels: ', senti_labels.shape)
        senti_labels = senti_labels[:, :, :-1]
        non_attr = np.zeros((attr_labels.shape[0],1),dtype='float32')
        non_attr_senti = np.tile(non_attr,reps=[1,3])
        non_attr_senti = np.expand_dims(non_attr_senti,axis=1)
        senti_labels = np.concatenate([senti_labels,non_attr_senti],axis=1)
        attr_labels = attr_labels[:top_k]
        senti_labels = senti_labels[:top_k]
        sentence = sentence[:top_k]
        print('senti labels shape: %s'%str(senti_labels.shape))
        print('attr labels shape: %s'%str(attr_labels.shape))
        print('sentence shape: %s'%str(sentence.shape))
    with open(outfile,'wb') as f:
        pickle.dump((attr_labels, senti_labels, sentence),f)
    print('test success')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk',type=int,default=None)
    parser.add_argument('--mod',type=str,default='fine')
    args = parser.parse_args()
    top_k = args.topk
    mod = args.mod

    path = {'coarse_train_in':'/datastore/liu121/sentidata2/data/aic2018/coarse_data_backup/train_coarse.pkl',
            'coarse_train_out':'/datastore/liu121/sentidata2/data/aic2018/coarse_data/train_coarse.pkl',
            'coarse_test_in':'/datastore/liu121/sentidata2/data/aic2018/coarse_data_backup/dev_coarse.pkl',
            'coarse_test_out':'/datastore/liu121/sentidata2/data/aic2018/coarse_data/dev_coarse.pkl',
            'fine_train_in':'/datastore/liu121/sentidata2/data/aic2018/fine_data_backup/train_fine.pkl',
            'fine_train_out':'/datastore/liu121/sentidata2/data/aic2018/fine_data/train_fine.pkl',
            'fine_test_in':'/datastore/liu121/sentidata2/data/aic2018/fine_data_backup/dev_fine.pkl',
            'fine_test_out':'/datastore/liu121/sentidata2/data/aic2018/fine_data/dev_fine.pkl',
           }
    if mod == 'coarse':
        train(path['coarse_train_in'],path['coarse_train_out'],top_k)
        test(path['coarse_test_in'],path['coarse_test_out'], top_k)
    else:
        train(path['fine_train_in'],path['fine_train_out'], top_k)
        test(path['fine_test_in'],path['fine_test_out'], top_k)