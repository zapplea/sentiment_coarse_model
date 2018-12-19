import pickle
import argparse
import numpy as np
np.set_printoptions(threshold=np.inf)

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
            write(outfile,key+'_'+str(data[key].shape)+':\n'+str(data[key])+'\n')
            outfile.flush()
            print(key)
    # TODO: whether the nan is caused by non-attribute.
    # TODO: check each operation, to see whether they get the expected value.
    # TODO: whehter it is caused by softmax loss when all label is 0
    # TODO: two ways to check 1. change attr pred labe to Y_att in joint 2. eliminate step2, only preserve step1 and 3.
    # TODO: cancel the others just presever the one which lead to nan at first
    # TODO: the "senti_W_attention" get nan at first, so let's check which value lead to this.
    # highlight: batch No. 75 lead to nan

def sentiment_attention(H, W, mask,):
    """
    :param H: shape = (batch size, number of words, lstm cell size)
    :param W: shape = (3*attribute numbers + 3,number of sentiment prototypes, lstm cell size). 3*attribute numbers is
    3 sentiment for each attributes; 3 is sentiment for non-attribute entity, it only has normal sentiment, not attribute
    specific sentiment.
    :param mask: mask to eliminate influence of 0; (3*attributes number+3, number of sentiment expression prototypes)
    :return: shape = (batch size,number of words, 3+3*attributes number, number of sentiment prototypes).
    """
    # # H.shape = (batch size, words num, 3+3*attributes number, word dim)
    # H = tf.tile(tf.expand_dims(H, axis=2), multiples=[1, 1, self.nn_config['sentiment_num'] * self.nn_config[
    #     'attributes_num'] + self.nn_config['sentiment_num'], 1])
    # # H.shape = (batch size, words num, 3+3*attributes number, sentiment prototypes, word dim)
    # H = tf.tile(tf.expand_dims(H, axis=3), multiples=[1, 1, 1, self.nn_config['normal_senti_prototype_num'] * self.nn_config['sentiment_num'] +
    #                                                   self.nn_config['attribute_senti_prototype_num'] *
    #                                                   self.nn_config['attributes_num'],
    #                                                   1])

    # H.shape = (batch size, number of words, lstm cell size)
    # W.shape = (3*attribute numbers + 3,number of sentiment prototypes, lstm cell size)
    # temp.shape = (batch size, words num, 3+3*attributes number, sentiment prototypes num)
    print('H shape: ',H.shape)
    print('W shape: ',W.shape)
    print('H is inf: ',np.any(np.isinf(H)))
    print('W is inf: ',np.any(np.isinf(W)))
    check_temp1 = np.tensordot(H, W,axes=[[-1],[-1]])
    for i in range(check_temp1.shape[0]):
        for j in range(check_temp1.shape[1]):
            for m in range(check_temp1.shape[2]):
                for l in range(check_temp1.shape[3]):
                    if np.isinf(np.exp(check_temp1[i,j,m,l])):
                        print('check_temp1 inf: ',check_temp1[i,j,m,l])
                        print(np.exp(check_temp1[i,j,m,l]))
                        print(np.exp(5))
                        exit()
    check_temp = np.exp(check_temp1)
    print('check temp is inf: ',np.any(np.isinf(check_temp)))
    temp = np.multiply(mask, np.exp(np.tensordot(H, W,axes=[[-1],[-1]])))

    # denominator.shape = (batch size, words num, 3+3*attributes number, 1)
    denominator = np.sum(temp, axis=3, keepdims=True)
    denominator = np.tile(denominator, reps=[1, 1, 1,
                                             4 * 3 +
                                             4 * 20])
    print('denominator isnan:',np.any(np.isnan(denominator)))
    for i in range(denominator.shape[0]):
        for j in range(denominator.shape[1]):
            for m in range(denominator.shape[2]):
                for l in range(denominator.shape[3]):
                    result = temp[i,j,m,l]/denominator[i,j,m,l]
                    if np.isnan(result):
                        print('[%d, %d, %d, %d]'%(i,j,m,l))
                        print('temp: ',temp[i,j,m,l])
                        print('denominator: ',denominator[i,j,m,l])
                        exit()
    attention = np.true_divide(temp, denominator)
    print('attention isnan: ',np.any(np.isnan(attention)))
    print('attention shape: ',attention.shape)
    return attention

def analysis2(dic):
    H = dic['senti_H']
    mask = dic['extors_mask_mat']
    W = dic['senti_W']
    attention = sentiment_attention(H,W,mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod',type=str,default='new')
    args = parser.parse_args()
    newpkl_filePath = '/datastore/liu121/sentidata2/report/coarse_nn/newpkl_reg1e-05_lr0.001_mat5.info'
    anal_filePath = '/datastore/liu121/sentidata2/report/coarse_nn/analysis_reg1e-05_lr0.001_mat5.info'
    if args.mod == "new":
        new_pickle(anal_filePath,newpkl_filePath)
    elif args.mod == 'anal':
        dic = load(newpkl_filePath)
        print(dic.keys())
        for key in dic:
            out_filePath ='/datastore/liu121/sentidata2/report/coarse_nn/result_%s.txt'%key
            analysis(dic,key,out_filePath)
    elif args.mod == 'anal2':
        newpkl_filePath = '/home/yibing/report/12.18/newpkl_reg1e-05_lr0.001_mat5.info'
        dic = load(newpkl_filePath)
        print('length of dic: %d'%len(dic))
        for key in dic:
            data = dic[key]
            analysis2(data)