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
        print('senti labels shape: %s'%repr(senti_labels.shape))
        print('attr labels shape: %s'%repr(attr_labels.shape))
        print('sentence shape: %s'%repr(sentence.shape))
    with open(outfile,'wb') as f:
        pickle.dump((attr_labels, senti_labels, sentence),f)
    print('test success')

def few_shot(infile,outfile, k_shot,mod):
    print('%s:'%mod)
    with open(infile,'rb') as f:
        if mod == 'train':
            attribute_dic, word_dic, attr_labels, senti_labels, sentences, word_embed = pickle.load(f)
        else:
            attr_labels, senti_labels, sentences = pickle.load(f)
        print('shape of sentences: ',sentences.shape)
        print('shape of attributes: ', attr_labels.shape)
        print('shape of senti labels: ', senti_labels.shape)

        non_attr = np.zeros((attr_labels.shape[0],1),dtype='float32')
        non_attr_senti = np.tile(non_attr,reps=[1,3])
        non_attr_senti = np.expand_dims(non_attr_senti,axis=1)
        senti_labels = np.concatenate([senti_labels,non_attr_senti],axis=1)
        shotted_attr_labels = []
        shotted_senti_labels = []
        shotted_sentences = []
        freq = []
        for i in range(20):
            freq.append(0)
        for attr,senti,sentence in zip(attr_labels,senti_labels,sentences):
            for i in range(20):
                if attr[i] ==1:
                    freq[i]+=1
            if np.any(np.less_equal(freq,k_shot)):
                shotted_attr_labels.append(attr)
                shotted_senti_labels.append(senti)
                shotted_sentences.append(sentence)
            if np.all(np.greater(freq,k_shot)):
                break

        senti_labels = np.array(shotted_senti_labels,dtype='int32')
        attr_labels = np.array(shotted_attr_labels, dtype='int32')
        sentences = np.array(shotted_sentences,dtype='int32')
        print('freq:\n',str(freq))
        print('senti labels shape: ',senti_labels.shape)
        print('attr labels shape: ',attr_labels.shape)
        print('sentence shape: ',sentences.shape)
    with open(outfile,'wb') as f:
        if mod == 'train':
            pickle.dump((attribute_dic, word_dic, attr_labels, senti_labels, sentences, word_embed),f)
        else:
            pickle.dump((attr_labels, senti_labels, sentence), f)
    print('train successful\n')

def test_few_shot(infile,outfile, top_k):
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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trk',type=int,default=None)
    parser.add_argument('--tek', type=int, default=None)
    parser.add_argument('--trf',type=int, default=float('inf'))
    parser.add_argument('--tef',type = int, default=float('inf'))
    parser.add_argument('--mod',type=str,default='fine')
    args = parser.parse_args()
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
    # if mod == 'coarse':
    #     train(path['coarse_train_in'],path['coarse_train_out'],args.trk)
    #     test(path['coarse_test_in'],path['coarse_test_out'], args.tek)
    # else:
    #     fine_train(path['fine_train_in'],path['fine_train_out'], args.trk)
    #     fine_test(path['fine_test_in'],path['fine_test_out'], args.tek)
    if mod == 'coarse':
        few_shot(path['coarse_train_in'],path['coarse_train_out'],args.trf,mod='train')
        few_shot(path['coarse_test_in'],path['coarse_test_out'], args.tef, mod='test')
    else:
        few_shot(path['fine_train_in'], path['fine_train_out'], args.trf, mod='train')
        few_shot(path['fine_test_in'], path['fine_test_out'], args.tef, mod='test')