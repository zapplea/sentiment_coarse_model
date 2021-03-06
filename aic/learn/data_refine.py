import pickle
import numpy as np
import argparse


def arrange(infile,outfile, bottom, up, mod):
    with open(infile,'rb') as f:
        if mod == 'train':
            print('train:')
            attribute_dic, word_dic, attr_labels, senti_labels, sentence, word_embed = pickle.load(f)
        else:
            print('test:')
            attr_labels, senti_labels, sentence = pickle.load(f)
        print('shape of sentence: ',sentence.shape)
        print('shape of attributes: ', attr_labels.shape)
        print('shape of senti labels: ', senti_labels.shape)

        non_attr = np.zeros((attr_labels.shape[0],1),dtype='float32')
        non_attr_senti = np.tile(non_attr,reps=[1,3])
        non_attr_senti = np.expand_dims(non_attr_senti,axis=1)
        senti_labels = np.concatenate([senti_labels,non_attr_senti],axis=1)
        attr_labels=attr_labels[bottom:up]
        senti_labels=senti_labels[bottom:up]
        sentence=sentence[bottom:up]
        print('senti labels shape: ',senti_labels.shape)
        print('attr labels shape: ',attr_labels.shape)
        print('sentence shape: ',sentence.shape)
    with open(outfile,'wb') as f:
        if mod == 'train':
            pickle.dump((attribute_dic, word_dic, attr_labels, senti_labels, sentence, word_embed),f)
            print('train successful\n')
        else:
            pickle.dump((attr_labels, senti_labels, sentence),f)
            print('test successful\n')

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
            if np.any(np.logical_and(np.equal(attr,1),np.less_equal(freq,k_shot))):
                shotted_attr_labels.append(attr)
                shotted_senti_labels.append(senti)
                shotted_sentences.append(sentence)
                for i in range(20):
                    if attr[i] == 1:
                        freq[i] += 1
            if np.all(np.greater(freq,k_shot)):
                break

        senti_labels = np.array(shotted_senti_labels,dtype='int32')
        attr_labels = np.array(shotted_attr_labels, dtype='int32')
        sentences = np.array(shotted_sentences,dtype='int32')
        print('freq:\n%s\n%s'%(str(np.sum(freq)),str(freq)))
        print('senti labels shape: ',senti_labels.shape)
        print('attr labels shape: ',attr_labels.shape)
        print('sentence shape: ',sentences.shape)
    with open(outfile,'wb') as f:
        if mod == 'train':
            pickle.dump((attribute_dic, word_dic, attr_labels, senti_labels, sentences, word_embed),f)
        else:
            pickle.dump((attr_labels, senti_labels, sentences), f)
    print('train successful\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trb',type=int,default=0)
    parser.add_argument('--tru', type=int, default=None)
    parser.add_argument('--teb', type=int, default=0)
    parser.add_argument('--teu', type=int, default=None)

    parser.add_argument('--trf',type=int, default=float('inf'))
    parser.add_argument('--tef',type = int, default=float('inf'))

    parser.add_argument('--cmod',type=str,default='coarse')
    parser.add_argument('--fmod',type=str,default='few_shot')
    args = parser.parse_args()

    path = {'coarse_train_in':'/datastore/liu121/sentidata2/data/aic2018/coarse_data_backup/train_coarse.pkl',
            'coarse_train_out':'/datastore/liu121/sentidata2/data/aic2018/coarse_data/train_coarse.pkl',
            'coarse_test_in':'/datastore/liu121/sentidata2/data/aic2018/coarse_data_backup/dev_coarse.pkl',
            'coarse_test_out':'/datastore/liu121/sentidata2/data/aic2018/coarse_data/dev_coarse.pkl',
            'fine_train_in':'/datastore/liu121/sentidata2/data/aic2018/fine_data_backup/train_fine.pkl',
            'fine_train_out':'/datastore/liu121/sentidata2/data/aic2018/fine_data/train_fine.pkl',
            'fine_test_in':'/datastore/liu121/sentidata2/data/aic2018/fine_data_backup/dev_fine.pkl',
            'fine_test_out':'/datastore/liu121/sentidata2/data/aic2018/fine_data/dev_fine.pkl',
           }

    if args.fmod == 'few_shot':
        if args.cmod == 'coarse':
            few_shot(path['coarse_train_in'], path['coarse_train_out'], args.trf, mod='train')
            few_shot(path['coarse_test_in'], path['coarse_test_out'], args.tef, mod='test')
        else:
            few_shot(path['fine_train_in'], path['fine_train_out'], args.trf, mod='train')
            few_shot(path['fine_test_in'], path['fine_test_out'], args.tef, mod='test')
    else:
        if args.cmod == 'coarse':
            arrange(path['coarse_train_in'],path['coarse_train_out'],args.trb,args.tru,mod='train')
            arrange(path['coarse_test_in'],path['coarse_test_out'], args.teb, args.teu, mod='test')
        else:
            arrange(path['fine_train_in'],path['fine_train_out'], args.trb,args.tru,mod='train')
            arrange(path['fine_test_in'],path['fine_test_out'], args.teb,args.teu,mod='test')

