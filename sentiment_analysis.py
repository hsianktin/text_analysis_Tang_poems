# Analysis based on gensim-Google w2v.
# tensorflow r2.0 is required. Or Keras is required.
import sys
from multiprocessing.dummy import Pool as ThreadPool
import tensorflow as tf  # **specifically, please use tensorflow 1.x**
from collections import defaultdict
import pandas as pd
import gensim.models
import string
import time
import csv
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import tempfile
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split

#from sklearn import train_test_split
batch_size = 20
file = open("Tang_poems_utf_8.txt")
max_len = 141
w2vmodel = gensim.models.Word2Vec.load('/tmp/mymodel')
embedding_dim = w2vmodel.vector_size
text = file.read()
file.close()
# split and create regular poems

pattern = u'卷.[0-9]+'
poems = re.split(pattern, text)[1:-1]
regular_sentences = []

# Exclude Non chinese_characters
chinese_punc = u'！|？|｡|。|＂|＃|＄|％|＆|＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|\
    ＠|［|＼|］|＾|＿|｀|｛|｜|｝|～|｟|｠|｢|｣|､|、|〃|》|「|」|『|』|【|】|〔|〕|〖|\
    〗|〘|〙|〚|〛|〜|〝|〞|〟|〰|〾|〿|–|—|‘|’|‛|“|”|„|‟|…|‧|﹏||《|□|\n'

other_non_cn = u'[0-9a-zA-Z]|\n'
exclude = set(string.punctuation)
stopwords = u'而|何|乎|乃|其|且|若|所|为|焉|以|因|于|与|也|则|者|之|不|自|得|一|来|去|无|可|是|已|此|的|上|中|兮|三'
re_auxiliary_words = re.compile("|".join([chinese_punc]))

for poem in poems:
    tmp_poem = poem.strip('\n\u3000\u3000◎')
    tmp_poem = re.split(chinese_punc, tmp_poem.replace('\u3000\u3000', ''))
    for j in range(1, len(tmp_poem)):
        regular_sentences.append(tmp_poem[j])

# %%
# Before network training, the first thing to do is to create a data set with labels.
# Despite that test sets should be iid samples of overall data, we choose those poems whose sentiment is clearly demonstrated by the sentimental words they use.
sentiments = ['思', '忧', '悲', '孤', '壮', '乐']
sent_tags = {
    ch: [x[0] for x in w2vmodel.most_similar(ch, topn=5)]
    for ch in sentiments
}
sent_tags['壮'].append('酒')
tagged_set = dict()
tagged_poems = []
poem_tags = []
for poem in regular_sentences:
    tmp = list()
    l = 0
    for ch in sentiments:
        m = len(set(poem).intersection(set(sent_tags[ch])))
        if m > l:
            l = m
            tag = ch
    if l > 0 and len(poem) < max_len and len(poem) > 4:
        tagged_set[poem] = tag
        poem_tags.append(tag)
        tagged_poems.append(poem)
#%%
# due to the requirement of keras, we still need a one-hop encoding of words.
vocab_list = [word for word, Vocab in w2vmodel.wv.vocab.items()]
word_index = {"": 0}  # initialize "{word: token}" dict
word_vector = {}  # initialize "{word: vector}" dict

# initialize a large matrix storing all vectors. take care that we need an additional first row.
# rows = 1 + len(vocab_list), columns = embeding_dim
embedding_matrix = np.zeros((len(vocab_list) + 1, w2vmodel.vector_size))
# fill in word_index and word_vector, as well as embedding_matrix
for i in range(len(vocab_list)):
    word = vocab_list[i]
    word_index[word] = i + 1
    word_vector[word] = w2vmodel.wv[word]
    embedding_matrix[i + 1] = w2vmodel.wv[word]

#%%
# Unify sentence length and other data preprocessing
x_data = []
y_data = []
count_y = {ch: 0 for ch in sentiments}
senti_class = dict()
for i in range(len(sentiments)):
    a = [0, 0, 0, 0, 0, 0]
    a[i] = 1
    senti_class[sentiments[i]] = np.array(a)
max_len = 0

for i in range(len(tagged_poems)):
    x_variable = []
    x = tagged_poems[i]
    y = poem_tags[i]
    for word in x:
        try:
            x_variable.append(word_index[word])
        except:
            x_variable.append(0)
    while len(x_variable) < max_len:
        x_variable.append(0)
    x_data.append(np.array(x_variable))
    y_data.append(senti_class[y])
    count_y[y] += 1

print(count_y)
x_data = np.array(x_data)
y_data = np.array(y_data)
x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                    y_data,
                                                    test_size=0.13,
                                                    random_state=42)


# %%
# Define network structure
# The Sequential model is a linear stack of layers.
# You can create a Sequential model by passing a list of layer instances to the constructor:
def create_model():
    model = Sequential()  # extensible architecher
    model.add(
        Embedding(len(embedding_matrix),
                  embedding_dim,
                  weights=[embedding_matrix],
                  input_length=max_len)
    )  # embedding accepts np.array input, and it converts a_i..j \in integers to [a_i...j_1,...a_i...j_d] the rows in embedding matrix
    # model.add(Conv1D(
    #     filters=64,
    #     kernel_size=3,
    #     activation='sigmoid',
    # ))

    model.add(LSTM(6, dropout=0.1, recurrent_dropout=0.2))
    model.add(Dense(6, activation='sigmoid'))
    print('compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


lstm_model = create_model()
lstm_model.summary()
print(u'training...')
lstm_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
print(u"evaluating...")
score, acc = lstm_model.evaluate(x_test, y_test)
print(u"acc=" + str(acc))
# %%
# from random import sample
# slices = sample(regular_sentences, 1000)
# regular_slices = []
# for poem in slices:
#     if len(poem) < 141 and len(poem) > 4:
#         regular_slices.append(poem)
# x_non_senti = []
# for x in regular_slices:
#     x_variable = []
#     for word in x:
#         try:
#             x_variable.append(word_index[word])
#         except:
#             x_variable.append(0)
#     while len(x_variable) < max_len:
#         x_variable.append(0)
#     x_non_senti.append(np.array(x_variable))
# x_non_senti = np.array(x_non_senti)
# regular_slices_senti = lstm_model.predict(x_non_senti)
# regular_slices_senti = np.ndarray.tolist(regular_slices_senti)
# for i in range(len(regular_slices_senti)):
#     print(u"诗的内容是：\n")
#     print(regular_slices[i])
#     # print(regular_slices_senti[i])
#     print(u"\n情感分析为：\n")
#     for j in range(len(sentiments)):
#         print(sentiments[j] + str(regular_slices_senti[i][j]) + "\n")


def sentiment_judge0(sentence):
    x_non_senti = []
    sentence = list(sentence)
    x_variable = []
    for x in sentence:
        try:
            x_variable.append(word_index[x])
        except:
            x_variable.append(0)
    while len(x_variable) < max_len:
        x_variable.append(0)
    x_non_senti.append(np.array(x_variable))
    x_non_senti = np.array(x_non_senti)
    regular_slices_senti = lstm_model.predict(x_non_senti)
    return regular_slices_senti


def sentiment_judge(sentence):
    x_non_senti = []
    sentence = list(sentence)
    x_variable = []
    for x in sentence:
        try:
            x_variable.append(word_index[x])
        except:
            x_variable.append(0)
    while len(x_variable) < max_len:
        x_variable.append(0)
    x_non_senti.append(np.array(x_variable))
    x_non_senti = np.array(x_non_senti)
    regular_slices_senti = lstm_model.predict(x_non_senti) - sentiment_judge0(
        "111")
    return regular_slices_senti


file = open("tang_samples.txt")
text = file.read()
file.close()
samples = text.split("\n")
for poem in samples:
    print(poem)
    sent_of_poem = np.ndarray.tolist(sentiment_judge(poem))
    for j in range(len(sentiments)):
        print(sentiments[j] + "：" + str(sent_of_poem[0][j]) + "\n")
# %%
# I made changes to the sentiments to implement a balanced analysis, keep the model unbiased.
# I also simplified the network structure.