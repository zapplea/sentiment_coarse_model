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
    for entity_name in comments:
        reviews=list(comments[entity_name])
        reviews.insert(0,['star_rating','review'])
        sheet=pe.Sheet(reviews)
        # f=open('/home/yibing/Documents/JD_op/coarse_label/'+entity_name+'.csv','w+')
        # f.close()
        sheet.save_as('/home/yibing/Documents/JD_op/coarse_label/'+entity_name+'.csv')


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

if __name__=="__main__":
    load()