import re


def find(sent):
    ls=sent.split('&')
    attribute=[]
    for s in ls:
        entity_senti=re.findall(r"(.*#-1|.*#0|.*#1)",s)
        if len(entity_senti)==0:
            continue
        else:
            entity,senti=tuple(entity_senti[0].split('#'))
            attribute.append((entity,senti))
