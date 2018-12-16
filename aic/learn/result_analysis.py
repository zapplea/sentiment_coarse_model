import pickle

def new_pickle(in_filePath,out_filePath):
    dic = {}
    with open(in_filePath,'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                count = 0
                for key in data:
                    ls = key.split()
                    key = ' '.join([ls[0],ls[1],str(count)])
                    print(key)
                    dic[key]=data[key]
                    count+=1
            except Exception:
                break
    with open(out_filePath,'wb') as f:
        pickle.dump(dic,f)

def load(filePath):
    with open(filePath,'rb') as f:
        data = pickle.load(f)
        return data

def write(outfile,info):
    outfile.write(str(info))

def analysis(dic,key,out_filePath):
    with open(out_filePath,'w+') as outfile:
        data = dic[key]
        for key in data:
            write(outfile,key+':\n'+str(data[key])+'\n')
    # TODO: whether the nan is caused by non-attribute.
    # TODO: check each operation, to see whether they get the expected value.
    # TODO: whehter it is caused by softmax loss when all label is 0
    # TODO: two ways to check 1. change attr pred labe to Y_att in joint 2. eliminate step2, only preserve step1 and 3.

if __name__ == "__main__":
    newpkl_filePath = '/datastore/liu121/sentidata2/report/coarse_nn/newpkl_reg1e-06_lr0.001_mat5.info'
    anal_filePath = '/datastore/liu121/sentidata2/report/coarse_nn/analysis_reg1e-06_lr0.001_mat5.info'
    new_pickle(anal_filePath,newpkl_filePath)
    # dic = load(newpkl_filePath)
    # s = '%s epoch: %d'
    # keys = [s%('joint',0),s%('joint',1),s%('joint',2),s%('joint',3)]
    # for key in keys:
    #     ls = key.split(' ')
    #     out_filePath ='/datastore/liu121/sentidata2/report/coarse_nn/result_%s.txt'%'_'.join([ls[0],ls[2]])
    #     analysis(dic,key,out_filePath)