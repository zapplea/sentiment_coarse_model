import pyexcel as pe
import numpy as np
import os
import jieba

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

    aspect={'物流':[('物流'),('发货'),('到货'),('送货'),('快递'),('配送')],
            '正品':[('正品'),('真品'),('正货')],
            '价格低':[('性价比',),('便宜')],
            '价格高':[('贵')],
            '1':[('杂音',)],
            '2':[('听','不清',),('没','声音',),('贵')],
            '3':[('屏幕','失','灵'),('屏幕','不','灵敏',)],
            '4':[('反应',)],
            '5':[('网速',)],
            '翻新机':[('翻新机',)],
            '发热':[('烫',),('发热',)],
            '充电':[('充','电')]}
    keywords=[('真品','正品','行货'),('假货','水货'  ),('杂音','听不见','听不到','听不清','没声音'),('优惠','便宜','不贵','性价比'),('运行速度',),('漂亮',)]
    count={}
    for t in comments:
        remarks[t]=[]
        for record in comments[t]:
            star=record[0]
            review=record[1]
            # review=jieba.cut(review)
            for keys in keywords:
                key=keys[0]
                for word in keys:
                    if word in review:
                        if key in count:
                            count[key]+=1
                        else:
                            count[key]=1
                        break
    print(count)

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

if __name__=="__main__":
    load()