# file import
import sys
import tensorflow as tf  # **specifically, please use tensorflow 1.x**
from collections import defaultdict
import seaborn as sns
import pandas as pd
import string

import time
import csv
import os
import matplotlib.font_manager as fm
import numpy as np
import re
import matplotlib.pyplot as plt
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
re_auxiliary_words = re.compile(
    "|".join([chinese_punc, other_non_cn, stopwords]))

for poem in poems:
    tmp_poem = poem.strip('\n\u3000\u3000◎')
    tmp_poem = tmp_poem.replace('\u3000\u3000', '').split('\n')
    regular_title.append(tmp_poem[0])
    regular_poems.append('\n'.join(tmp_poem[1:]))
    tmp_poem = ''.join(ch for ch in list(
        ''.join(tmp_poem[1:])) if ch not in exclude)
    corpus.append(list(re_auxiliary_words.sub('', tmp_poem)))


corpus = corpus[0:N]
# develop based on this part
# create a vocabulary
words = []
for poem in corpus:
    for word in poem:
        words.append(word)
words = set(words)
# create map between word and integers
word2int = {}
int2word = {}
vocab_size = len(words)  # gives the total number of unique words
for i, word in enumerate(words):
    word2int[word] = i
    int2word[i] = word
sentences = corpus
# capture each word and their neighborhood
data = []
ncount = 0
WINDOW_SIZE = 2
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0):min(word_index + WINDOW_SIZE, len(sentence) + 1)]:
            if nb_word != word:
                ncount += 1
                data.append([word, nb_word])


def unit_vectorization(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp


vocab_size = len(words)
x_train = []
y_train = []
for data_word in data:
    x_train.append(unit_vectorization(word2int[data_word[0]], vocab_size))
# convert to nparray
x_train = np.asarray(x_train)  # , dtype=np.int16)
# x and y are initialized separately to save memory
for data_word in data:
    y_train.append(unit_vectorization(word2int[data_word[1]], vocab_size))
y_train = np.asarray(y_train)  # , dtype=np.int16)

# make the tensorflow model
x = tf.placeholder(tf.float32, [None, vocab_size], name="x")
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size), name="y_label")

EMBEDDING_DIM = 4
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]), name="W1")
# noise ### this line claims a variable of tf type which is essentially a 1x5 matrix
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]), name="b1")
hidden_representation = tf.add(tf.matmul(x, W1), b1)
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]), name="W2")
b2 = tf.Variable(tf.random_normal([vocab_size]), name="b2")
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation, W2), b2))
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)  # make sure you do this!
print('Preparation Done')
# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(
    y_label * tf.log(prediction), reduction_indices=[1]))
# define the training step:
train_step = tf.train.GradientDescentOptimizer(
    0.1).minimize(cross_entropy_loss)
print('Start calculating')
# train for n_iter iterations
n_iters = 10000
saver = tf.train.Saver()
checkpoint_path = "/W2VMODEL/cp.ckpt"
count = 0
for _ in range(n_iters):
    count += 1
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    if count % 10 == 1:
        print(str(count)+'th iteration, loss is : ',
              sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))
    if count % 100 == 0:
        save_path = saver.save(sess, checkpoint_path)

# output
vectors = sess.run(W1 + b1)


def word_vec(word):
    v_w = vectors(word2int[word])
    return v_w


def vec_sim(vec, top_n):
    word_sim = {}
    for i in range(vectors.shape[0]):
        v_w2 = vectors[i]
        theta_sum = np.dot(vec, v_w2)
        theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
        theta = theta_sum / theta_den
        word = int2word[i]
        word_sim[word] = theta
    word_sorted = sorted(word_sim.items(), key=lambda x: x[1], reverse=True)
    for word, sim in word_sorted[1:top_n+1]:
        print(word, sim)
    pass


def word_sim(word, top_n):
    w1_index = word2int[word]
    v_w1 = vectors[w1_index]
    word_sim = {}
    for i in range(vectors.shape[0]):
        v_w2 = vectors[i]
        theta_num = np.dot(v_w1, v_w2)
        theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
        theta = theta_num / theta_den

        word = int2word[i]
        word_sim[word] = theta

    words_sorted = sorted(word_sim.items(), key=lambda x: x[1], reverse=True)

    for word, sim in words_sorted[1:top_n+1]:
        print(word, sim)

    pass


filename_w2v = './output/w2v_matrix.txt'
with open(filename_w2v, 'w') as f:
    f.write("字\t对应向量\t训练样本数"+str(N))
    for mu in words:
        f.write("\n"+x+"\t"+str(word_vec(mu)))
filename_w2v_sample = "./output/w2v_sample.txt"
sample = ['思', '悲', '忧', '愁', '怒', '惧', '乐']
with open(filename_w2v_sample, 'w') as f:
    for mu in sample:
        f.write(mu+": " + "、".join([y[0] for y in word_sim(mu, 7)]))

print('Files are saved')
