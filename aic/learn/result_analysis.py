import pickle

def load(filePath):
    dic = {}
    with open(filePath,'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                for key in data:
                    dic[key]=data[key]
            except Exception:
                break
    return dic

def write(outfile,info):
    outfile.write(str(info))

def analysis(dic,key,outfile):
    data = dic[key]
    for key in data:
        write(outfile,key+':\n'+str(data[key])+'\n')
    # TODO: whether the nan is caused by non-attribute.
    # TODO: check each operation, to see whether they get the expected value.
    # TODO: whehter it is caused by softmax loss when all label is 0
    # TODO: two ways to check 1. change attr pred labe to Y_att in joint 2. eliminate step2, only preserve step1 and 3.

if __name__ == "__main__":
    dic = load('/home/yibing/report/12.11/analysis_reg1e-05_lr0.001_mat5.info')
    with open('/home/yibing/report/12.11/result.txt','w+') as f:
        analysis(dic,'%s epoch: %d'%('senti',0),f)