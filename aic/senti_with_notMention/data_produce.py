import pickle
import numpy as np

with open('/datastore/liu121/sentidata2/data/aic2018/coarse_data/train_coarse.pkl','rb') as f:
    attribute_dic, word_dic, attr_labels, senti_labels, sentence, word_embed = pickle.load(f)
    print(senti_labels.shape)
    print(attr_labels.shape)
    non_attr = np.zeros((attr_labels.shape[0],1),dtype='float32')
    for i in range(attr_labels.shape[0]):
        al = attr_labels[i]
        if np.sum(al) == 0:
            pass

# with open('/datastore/liu121/sentidata2/expdata/aic2018/coarse_data/dev_fine.pkl','wb') as f:
#     pickle.dump(data,f)