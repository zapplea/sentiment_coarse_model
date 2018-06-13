import os

pathes=['/home/yibing/Documents/hotel_review/ChnSentiCorp_htl_ba_2000/neg',
      '/home/yibing/Documents/hotel_review/ChnSentiCorp_htl_ba_2000/pos',
      '/home/yibing/Documents/hotel_review/ChnSentiCorp_htl_ba_4000/neg',
      '/home/yibing/Documents/hotel_review/ChnSentiCorp_htl_ba_4000/pos',
      '/home/yibing/Documents/hotel_review/ChnSentiCorp_htl_ba_6000/neg',
      '/home/yibing/Documents/hotel_review/ChnSentiCorp_htl_ba_6000/pos',
      '/home/yibing/Documents/hotel_review/ChnSentiCorp_htl_unba_10000/neg',
      '/home/yibing/Documents/hotel_review/ChnSentiCorp_htl_unba_10000/pos']

texts={'neg':[],'pos':[]}
count = 0
for path in pathes:
    fps=os.listdir(path)
    for fp in fps:
        txt = []
        index = path.split('/')[-1]
        with open(path+'/'+fp,'rb') as file:
            for line in file:
                try:
                    txt.append(line.decode('gb2312'))
                except Exception:
                    #print(path + '/' + fp)
                    count+=1
                    continue
        texts[index].append(tuple(txt))
neg = texts['neg']
pos = texts['pos']
for i in range(len(neg)):
    org= neg[i]
    if i+2< len(neg):
        for j in range(i+1,len(neg)):
            new = neg[j]
            if neg[i] == neg[j]:
                print(neg[i])
                input()
print(count)
print(len(neg)+len(pos))
