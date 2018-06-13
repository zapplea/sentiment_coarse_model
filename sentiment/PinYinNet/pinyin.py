"""
Implemention for map single word to pinyin or an word document to pinyin document
Method contains normal(with tone) and lazy(without tone)

"""


from pypinyin import pinyin, lazy_pinyin, Style
import pandas as pd
import numpy as np


def word2pinyin(word):
    """
    both word and returned pinyin is utf-8 encode, 
    Given a word return PinYin 
    """
    pinyin_list = pinyin(word)
    word_pinyin = u''
    for i in range(len(pinyin_list)):
        word_pinyin += pinyin_list[i][0]
    return word_pinyin

def word2lazyPinYin(word):
    # PinYin with lazy style
    pinyin_list = lazy_pinyin(word)
    word_pinyin = u''
    for i in range(len(pinyin_list)):
        word_pinyin += pinyin_list[i]
    return word_pinyin

def tokText2tokPinYin(tok,method):
    """
    Given text of token word, return texts of PinYin
    method can be either "normal" or "lazy" style
    """
    pinyin_text=[]
    for i in range(len(tok)):
        pinyin_sentence = []
        for j in range(len(tok[i])):
            if method == "normal":
                pinyin_sentence.append(word2pinyin(tok[i][j]))
            elif method == "lazy":
                pinyin_sentence.append(word2lazyPinYin(tok[i][j]))
        pinyin_text.append(pinyin_sentence)
    return pinyin_text