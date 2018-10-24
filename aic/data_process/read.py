import pickle
import numpy as np
with open('/datastore/liu121/sentidata2/expdata/aic2018/fine_data/dev_fine.pkl','rb') as f:
    data = pickle.load(f)
    sentence = []
    for s in data[2]:
        sentence.append(s)
    sentence=np.array(sentence,dtype='int32')
    data=(data[0],data[1],sentence)
    print(data[0].shape,' ',data[1].shape,' ',data[2].shape)
