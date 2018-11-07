import pickle
import numpy as np
with open('/datastore/liu121/sentidata2/data/aic2018/coarse_data/train_coarse.pkl','rb') as f:
    attribute_dic, word_dic, attr_labels, senti_labels, sentence, word_embed = pickle.load(f)
    print(senti_labels.shape)

# with open('/datastore/liu121/sentidata2/expdata/aic2018/coarse_data/dev_fine.pkl','wb') as f:
#     pickle.dump(data,f)