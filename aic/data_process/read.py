import pickle
import numpy as np

with open('/datastore/liu121/sentidata2/data/aic2018/coarse_data_backup/dev_coarse.pkl','rb') as f:
    attr_labels, senti_labels, sentence = pickle.load(f)
    print(senti_labels.shape)
    print(attr_labels.shape)
    non_attr = np.zeros((attr_labels.shape[0],1),dtype='float32')
    non_attr_senti = np.tile(non_attr,reps=[1,3])
    for i in range(attr_labels.shape[0]):
        al = attr_labels[i]
        if np.sum(al) == 0:
            non_attr_senti[i][1]=1
    print(non_attr_senti.shape)
    non_attr_senti = np.expand_dims(non_attr_senti,axis=1)
    print(non_attr_senti.shape)
    print(senti_labels[0])
    senti_labels = np.concatenate([senti_labels,non_attr_senti],axis=1)
    print(senti_labels.shape)
    print(senti_labels[0])

with open('/datastore/liu121/sentidata2/data/aic2018/coarse_data/dev_coarse.pkl','wb') as f:
    pickle.dump((attr_labels, senti_labels, sentence),f)

# with open('/datastore/liu121/sentidata2/expdata/aic2018/coarse_data/dev_fine.pkl','wb') as f:
#     pickle.dump(data,f)