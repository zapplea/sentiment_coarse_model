import numpy as np
import math
import operator
from pathlib import Path
import pickle

def __init__(datafeed_init):
    global data
    global embfp
    global m4mat
    global top_k
    global tablefp
    global contxt_file
    global with_padding
    with_padding=datafeed_init['with_padding']
    tablefp=datafeed_init['table']
    top_k = datafeed_init['top_k']
    pkl_file = datafeed_init['pkl_name']
    contxt_file=datafeed_init['contxt_file']
    pkl_check = Path(pkl_file)
    if not pkl_check.is_file():
        mention_type = extract_mentionType()  # {(lemma,lcontxt,rcontxt):t_id, ...}
        print('length of mention_type: ', str(len(mention_type)))
        data=map_words_to_id(mention_type)
        np.random.shuffle(data)
        outf = open(pkl_file, 'wb')
        pickle.dump(data, outf)
        outf.flush()
        outf.close()
    else:
        inf = open(pkl_file, 'rb')
        data = pickle.load(inf)
        np.random.shuffle(data)
        inf.close()

def extract_mentionType():
    """ 
    :return: {(lemma,lcontxt,rcontxt):t_id, ...}
    """
    # order t_id based on frequency
    f=open(contxt_file,'rb')
    queue=pickle.load(f) # [ [ [lemma,lcontxt,rcontxt],[t_ids] ], ...]
    freq={}
    for ls in queue:
        t_ids=ls[1]
        for t_id in t_ids:
            if t_id not in freq:
                freq[t_id]=1
            else:
                freq[t_id]+=1
    tid_freq=sorted(freq.items(),key=operator.itemgetter(1,0),reverse=True)

    # map old t_id to new t_id based on its frequency
    old_tid_to_new_tid={}
    top_k_type_set=set()
    for i in range(top_k):
        top_k_type_set.add(tid_freq[i][0])
        old_tid_to_new_tid[tid_freq[i][0]]=i
    print('length of top_k_type_set: ',str(len(top_k_type_set)))

    # delete
    dic={} # {(lemma,lcontxt,rcontxt):[tid,tid, ...]}
    for ls in queue:
        t_ids=ls[1]
        contxt=tuple(ls[0]) # (lemma,lcontxt,rcontxt)
        types=[]
        for t_id in t_ids:
            if t_id in top_k_type_set:
                types.append(old_tid_to_new_tid[t_id])
        if len(types)>0:
            dic[contxt]=types
    return dic


def map_words_to_id(mention_type):
    """
    :param mentionType: {(lemma,lcontxt,rcontxt):[t_id, ...], ...}
    :return: [([table ids],[type ids])]
    """
    table_file=open(tablefp,'rb')
    dictionary=pickle.load(table_file)
    table_file.close()
    data=[]
    for key in mention_type:
        t_id_list=mention_type[key]
        lemma=key[0]
        lcontxt=key[1]
        rcontxt=key[2]
        words=lcontxt.split()
        lemma_ls=lemma.split()
        while len(lemma_ls)<with_padding-4:
            lemma_ls.extend(['#PAD#'])
        words.extend(lemma_ls)
        words.extend(rcontxt.split())
        id=[]
        for word in words:
            id.append(dictionary[word])
        data.append((id,t_id_list))
    return data


def data_generator(mode, batch_num, batch_size, out_layer_dim):
    """
    This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch. 
    :param batch_size: int. the size of each batch
    :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
    """
    # [( emb_id,fname,row_index m_id,c_id,typeText)]
    if mode == 'train':
        # train_size = math.ceil(len(data) * 0.995)
        # data_temp = data[:train_size]
        data_temp = data[:]
    elif mode == 'test':
        # train_size = math.ceil(len(data)) - math.ceil(len(data) * 0.9)
        # data_temp = data[math.ceil(len(data) * 0.9):math.ceil(len(data))]
        data_temp = data[-1000:]

    if mode == 'train':
        train_size=len(data_temp)
        start = batch_num * batch_size % train_size
        end = (batch_num * batch_size + batch_size) % train_size
        if start < end:
            batches = data_temp[start:end]
        elif start >= end:
            batches = data_temp[start:]
            # if mode=="train":
            batches.extend(data_temp[0:end])
    else:
        batches = data_temp
    x = []
    y_ = []
    for batch in batches:
        x.append(np.array(batch[0],dtype='int32'))
        labels = np.array(batch[1], dtype='int32')
        one_hot = np.zeros(shape=out_layer_dim, dtype='float32')
        for pos in labels:
            one_hot[pos] = 1.0
        y_.append(one_hot)
    # during validation and test, to avoid errors are counted repeatedly,
    # we need to avoid the same data sended back repeately
    return (np.array(x), np.array(y_))


def table_generator():
    table_file=open(tablefp,'rb')
    dictionary=pickle.load(table_file)
    table=pickle.load(table_file)
    table_file.close()
    del dictionary
    return table

if __name__=="__main__":
    table_generator()