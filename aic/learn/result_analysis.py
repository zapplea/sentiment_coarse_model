import pickle
import argparse

def new_pickle(in_filePath,out_filePath):
    dic = {}
    with open(in_filePath,'rb') as f:
        count = 0
        while True:
            try:
                data = pickle.load(f)
                for key in data:
                    print(key)
                    dic[key]=data[key]
            except Exception:
                break
    print('load successfully')
    with open(out_filePath,'wb') as f:
        pickle.dump(dic,f)
        f.flush()
        print('dump successfully')

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
    # highlight: batch No. 75 lead to nan

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod',type=str,default='new')
    args = parser.parse_args()
    newpkl_filePath = '/datastore/liu121/sentidata2/report/coarse_nn/newpkl_reg1e-05_lr0.001_mat5.info'
    anal_filePath = '/datastore/liu121/sentidata2/report/coarse_nn/analysis_reg1e-05_lr0.001_mat5.info'
    if args.mod == "new":
        new_pickle(anal_filePath,newpkl_filePath)
    else:
        dic = load(newpkl_filePath)
        for key in dic:
            out_filePath ='/datastore/liu121/sentidata2/report/coarse_nn/result_%s.txt'%key
            analysis(dic,key,out_filePath)