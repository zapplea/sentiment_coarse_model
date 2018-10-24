import pickle

with open('/datastore/liu121/sentidata2/expdata/aic2018/fine_data/dev_fine.pkl','rb') as f:
    data = pickle.load(f)
    print(data[2].shape)