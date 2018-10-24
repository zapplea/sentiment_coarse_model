import pickle
import numpy as np
with open('/datastore/liu121/sentidata2/expdata/aic2018/fine_data/dev_fine.pkl','rb') as f:
    data = pickle.load(f)
    print(len(data))
    sentence = []
    for s in data[2]:
        sentence.append(s)
    sentence=np.array(sentence,dtype='int32')
    print(sentence.shape)