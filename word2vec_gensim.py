# gensim has inbuilt implementation for word2vec. It deals with memory consumption by randomly initiate an N-dimensional vector.
import sys
import tensorflow as tf  # **specifically, please use tensorflow 1.x**
from collections import defaultdict
import pandas as pd
import string
import gensim.models
import time
import csv
import os
import matplotlib.font_manager as fm
import numpy as np
import re
import matplotlib.pyplot as plt
import tempfile

file = open("Tang_poems_utf_8.txt")
text = file.read()
file.close()
# split and create regular poems

pattern = u'卷.[0-9]+'
poems = re.split(pattern, text)[1:-1]
regular_poems = []
regular_title = []
corpus = []
N = 500  # total number of samples

# Exclude Non chinese_characters
chinese_punc = u'！|？|｡|。|＂|＃|＄|％|＆|＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|\
    ＠|［|＼|］|＾|＿|｀|｛|｜|｝|～|｟|｠|｢|｣|､|、|〃|》|「|」|『|』|【|】|〔|〕|〖|\
    〗|〘|〙|〚|〛|〜|〝|〞|〟|〰|〾|〿|–|—|‘|’|‛|“|”|„|‟|…|‧|﹏||《|□'

other_non_cn = u'[0-9a-zA-Z]|\n'
exclude = set(string.punctuation)
stopwords = u'而|何|乎|乃|其|且|若|所|为|焉|以|因|于|与|也|则|者|之|不|自|得|一|来|去|无|可|是|已|此|的|上|中|兮|三'
re_auxiliary_words = re.compile("|".join([chinese_punc, other_non_cn]))

for poem in poems:
    tmp_poem = poem.strip('\n\u3000\u3000◎')
    tmp_poem = tmp_poem.replace('\u3000\u3000', '').split('\n')
    regular_title.append(tmp_poem[0])
    regular_poems.append('\n'.join(tmp_poem[1:]))
    tmp_poem = ''.join(ch for ch in list(''.join(tmp_poem[1:]))
                       if ch not in exclude)
    corpus.append(list(re_auxiliary_words.sub('', tmp_poem)))

model = gensim.models.Word2Vec(corpus, min_count=10, iter=100, workers=8)

filename_w2v_sample = "./output/w2v_sample.txt"
sample = ['思', '悲', '忧', '愁', '愤', '惧', '乐']
with open(filename_w2v_sample, 'w') as f:
    for mu in sample:
        f.write(mu + ": " + "、".join([y[0]
                                      for y in model.most_similar(mu)]) + "\n")
model.save('/tmp/mymodel')
print('Files are saved')