# This is a non-tensorflow dependent implementation of w2v network analysis. Because of this, it's slow.
# file import
from collections import defaultdict
import seaborn as sns
import pandas as pd
import time
import csv
import os
import matplotlib.font_manager as fm
import numpy as np
import re
file = open("Tang_poems_utf_8.txt")
text = file.read()
file.close()
# split and create regular poems

# ------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple word2vec from scratch with Python
#   2018-FEB
#
# ------------------------------------------------------------------------------+

# --- IMPORT DEPENDENCIES ------------------------------------------------------+


# --- CONSTANTS ----------------------------------------------------------------+


class word2vec():

    def __init__(self):
        self.n = settings['n']
        self.eta = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']
        pass

    # GENERATE TRAINING DATA
    def generate_training_data(self, settings, corpus):

        # GENERATE WORD COUNTS
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1

        self.v_count = len(word_counts.keys())

        # GENERATE LOOKUP DICTIONARIES
        self.words_list = sorted(list(word_counts.keys()), reverse=False)
        self.word_index = dict((word, i)
                               for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word)
                               for i, word in enumerate(self.words_list))

        training_data = []
        # CYCLE THROUGH EACH SENTENCE IN CORPUS
        for sentence in corpus:
            sent_len = len(sentence)

            # CYCLE THROUGH EACH WORD IN SENTENCE
            for i, word in enumerate(sentence):

                # w_target  = sentence[i]
                w_target = self.word2onehot(sentence[i])

                # CYCLE THROUGH CONTEXT WINDOW
                w_context = []
                for j in range(i-self.window, i+self.window+1):
                    if j != i and j <= sent_len-1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])
        return np.array(training_data)

    # SOFTMAX ACTIVATION FUNCTION

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # CONVERT WORD TO ONE HOT ENCODING

    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec

    # FORWARD PASS

    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u

    # BACKPROPAGATION

    def backprop(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        # UPDATE WEIGHTS
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)
        pass

    # TRAIN W2V model

    def train(self, training_data):
        # INITIALIZE WEIGHT MATRICES
        # embedding matrix
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.n))
        # context matrix
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v_count))

        # CYCLE THROUGH EACH EPOCH
        for i in range(0, self.epochs):

            self.loss = 0

            # CYCLE THROUGH EACH TRAINING SAMPLE
            for w_t, w_c in training_data:

                # FORWARD PASS
                y_pred, h, u = self.forward_pass(w_t)

                # CALCULATE ERROR
                EI = np.sum([np.subtract(y_pred, word)
                             for word in w_c], axis=0)

                # BACKPROPAGATION
                self.backprop(EI, h, w_t)

                # CALCULATE LOSS
                self.loss += - \
                    np.sum([u[word.index(1)] for word in w_c]) + \
                    len(w_c) * np.log(np.sum(np.exp(u)))
                # self.loss += -2*np.log(len(w_c)) -np.sum([u[word.index(1)] for word in w_c]) + (len(w_c) * np.log(np.sum(np.exp(u))))

            print('EPOCH:', i, 'LOSS:', self.loss)
        pass

    # input a word, returns a vector (if available)

    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    # input a vector, returns nearest word(s)
    def vec_sim(self, vec, top_n):

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(vec, v_w2)
            theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(
            word_sim.items(), key=lambda x: x[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)

        pass

    # input word, returns top [n] most similar words
    def word_sim(self, word, top_n):

        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(
            word_sim.items(), key=lambda x: x[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)

        pass
# ------------------------END OF WORD2VEC-------------------------------------
#


pattern = u'卷.[0-9]+'
poems = re.split(pattern, text)[1:]
regular_poems = []
regular_title = []
corpus = []
N = 100  # total number of samples
# stopwords correction Chinese marks -> unicode string
stopwords = re.compile(
    '而|何|乎|乃|其|且|然|若|所|为|焉|也|以|矣|于|之|则|者|与|欤|因|[\u002d|\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]|[0-9]|[a-zA-Z]')
)
for poem in poems:
    tmp_poem=poem.strip('\n\u3000\u3000◎')
    tmp_poem=tmp_poem.replace('\u3000\u3000', '').split('\n')
    regular_title.append(tmp_poem[0])
    regular_poems.append('\n'.join(tmp_poem[1:]))
    tmp_poem=''.join(tmp_poem[1:]).replace('\n', "").replace('。', '').replace('，', '').replace(
        '：', '').replace('；', '').replace('？', '').replace('！', '').replace('（[.*]*?）', '')
    corpus.append(list(stopwords.sub('', tmp_poem)))
    # Word frequency

settings={}
settings['n']=7                   # dimension of word embeddings
settings['window_size']=2         # context window +/- center word
settings['min_count']=0           # minimum word count
settings['epochs']=5000           # number of training epochs
# number of negative words to use during training
settings['neg_samp']=10
settings['learning_rate']=0.01    # learning rate
np.random.seed(0)                   # set the seed for reproducibility
w2v=word2vec()
# corpus=corpus
corpus=corpus[0:N]
training_data=w2v.generate_training_data(settings, corpus)
w2v.train(training_data)
filename_w2v='./output/w2v_matrix.txt'
with open(filename_w2v, 'w') as f:
    f.write("字\t对应向量\t训练样本数"+str(N))
    for x in w2v.words_list:
        f.write("\n"+x+"\t"+str(w2v.word_vec(x)))
filename_w2v_sample="./output/w2v_sample.txt"
sample=['思', '悲', '忧', '愁', '怒', '惧', '乐']
with open(filename_w2v_sample, 'w') as f:
    for x in sample:
        f.write(x+": " + "、".join([y[0] for y in w2v.word_sim(x, 7)]))
