import pyexcel as pe
import numpy as np
import os


def load():
    root="/home/yibing/Documents/JD"
    paths_dic=path_dig(root)

    comments={}
    count=0
    for t in paths_dic:
        comments[t]=[]
        paths=paths_dic[t]
        for file_path in paths:
            records=pe.iget_records(file_name=file_path)
            repeat_check=set()
            for record in records:
                if '评论等级' in record:
                    star=record['评论等级']
                else:
                    star=record['星级']
                review=record['内容']
                if (star,review) not in repeat_check:
                    repeat_check.add((star,review))
                    comments[t].append((star,review))
                    count+=1
        comments[t]=np.array(comments[t])
        np.random.shuffle(comments[t])
    print(count)
    remarks={}
    for t in comments:
        remarks[t]=[]
        for record in comments[t]:
            star=record[0]
            review=record[1]
            content = []
            for sent in sent_tokenize(review):
                content.append(sent)
            remarks[t].append((star,content))
    return remarks



def path_dig(root):
    file_path={}
    dirs=os.listdir(root)
    for dir in dirs:
        path=root+'/'+dir
        files=os.listdir(path)
        file_path[dir]=[]
        for file in files:
            file_path[dir].append(path+'/'+file)
    return file_path


def sent_tokenize(sentence):
    puns = frozenset(u'。！？')
    tmp = []
    count=0
    for ch in sentence:
        tmp.append(ch)
        if puns.__contains__(ch):
            if count+1<len(sentence):
                if not puns.__contains__(sentence[count+1]):
                    yield ''.join(tmp)
                    tmp = []
            else:
                yield ''.join(tmp)
                tmp = []
        count+=1
    if len(tmp)>0:
        yield ''.join(tmp)


def output(comments):
    """
    :param comments: {'iphone8':[(star,[sent1,sent2,...]),...]}
    :return: 
    """
    root='/home/yibing/Documents/JD_op/manual_label'
    for t in comments:
        path=root+'/'+t
        os.makedirs(path)
        file_count=0
        count=0
        fpath=path+'/'+str(file_count)+'.txt'
        f=open(fpath,'w+')
        for record in comments[t]:
            star=record[0]
            content=record[1]
            f.write(star+'\n')
            for sent in content:
                f.write(sent+'\n')
            f.write('[]\n')
            count+=1
            if count%100==0:
                f.close()
                file_count+=1
                fpath=path+'/'+str(file_count)+'.txt'
                f=open(fpath,'w+')
        f.close()



def word_split():
    pass

if __name__=="__main__":
    comments=load()
    output(comments)

    # test data generate
    # p=comments['iphone8']
    # np.random.shuffle(p)
    # f1=open('/home/yibing/Documents/JD_op/manual_label/p1.txt','w+')
    # f2=open('/home/yibing/Documents/JD_op/manual_label/p2.txt','w+')
    # for record in p[:60]:
    #     star = record[0]
    #     content = record[1]
    #     f1.write(star + '\n')
    #     for sent in content:
    #         f1.write(sent + '\n')
    #     f1.write('[]\n')
    # p2=[]
    # p2.extend(p[:30])
    # p2.extend(p[60:])
    # for record in p2[:60]:
    #     star = record[0]
    #     content = record[1]
    #     f2.write(star + '\n')
    #     for sent in content:
    #         f2.write(sent + '\n')
    #     f2.write('[]\n')






















