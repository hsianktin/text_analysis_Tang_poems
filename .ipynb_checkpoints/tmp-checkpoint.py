## file import
file = open("Tang_poems_utf_8.txt")
text = file.read()
file.close()
import matplotlib.pyplot as plt
## split and create regular poems
import re
import numpy as np
import matplotlib.font_manager as fm
import os
import csv
import time
import pandas as pd
import seaborn as sns
from collections import defaultdict

#------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple word2vec from scratch with Python
#   2018-FEB
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import numpy as np
import re
from collections import defaultdict

#--- CONSTANTS ----------------------------------------------------------------+


class word2vec():

    def __init__ (self):
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
        self.words_list = sorted(list(word_counts.keys()),reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        # CYCLE THROUGH EACH SENTENCE IN CORPUS
        for sentence in corpus:
            sent_len = len(sentence)

            # CYCLE THROUGH EACH WORD IN SENTENCE
            for i, word in enumerate(sentence):
                
                #w_target  = sentence[i]
                w_target = self.word2onehot(sentence[i])

                # CYCLE THROUGH CONTEXT WINDOW
                w_context = []
                for j in range(i-self.window, i+self.window+1):
                    if j!=i and j<=sent_len-1 and j>=0:
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
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.n))     # embedding matrix
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v_count))     # context matrix
        
        # CYCLE THROUGH EACH EPOCH
        for i in range(0, self.epochs):

            self.loss = 0

            # CYCLE THROUGH EACH TRAINING SAMPLE
            for w_t, w_c in training_data:

                # FORWARD PASS
                y_pred, h, u = self.forward_pass(w_t)
                
                # CALCULATE ERROR
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                # BACKPROPAGATION
                self.backprop(EI, h, w_t)

                # CALCULATE LOSS
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
                #self.loss += -2*np.log(len(w_c)) -np.sum([u[word.index(1)] for word in w_c]) + (len(w_c) * np.log(np.sum(np.exp(u))))
                
            print('EPOCH:',i, 'LOSS:', self.loss)
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

        words_sorted = sorted(word_sim.items(), key=lambda x:x[1], reverse=True)

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

        words_sorted = sorted(word_sim.items(), key=lambda  x:x[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)
            
        pass
# ------------------------END OF WORD2VEC-------------------------------------
#

# I'm using WSL2, thus in order to provide chinese support, I need to configure fonts myself
plt.rcParams['font.sans-serif']=['Taipei Sans TC Beta']
pattern = u'卷.[0-9]+'
poems = re.split(pattern, text)[1:]
regular_poems = []
regular_title = []
corpus = []
stopwords = re.compile(u'而|何|乎|乃|其|且|然|若|所|为|焉|也|以|矣|于|之|则|者|与|欤|因|-')
for poem in poems:
    tmp_poem = poem.strip('\n\u3000\u3000◎')
    tmp_poem = tmp_poem.replace('\u3000\u3000','').split('\n')
    regular_title.append(tmp_poem[0])
    regular_poems.append('\n'.join(tmp_poem[1:]))
    tmp_poem=''.join(tmp_poem[1:]).replace('\n',"").replace('。','').replace('，','').replace('：','').replace('；','').replace('？','').replace('！','').replace('（[.*]*?）','')
    corpus.append([stopwords.sub('',tmp_poem)])
    ## Word frequency
data = ("".join(regular_poems)).replace('\n',"").replace('。','').replace('，','').replace('：','').replace('；','').replace('？','').replace('！','').replace('（[.*]*?）','')
data = stopwords.sub('',data) # sample calculation; reduced size
data = list(data)
wordlist = set(data)
diction = {}
n = 0
N = len(wordlist)
for word in wordlist:
    n += 1
    diction[word]=data.count(word)*100/N
    if n % 100 == 0:
        print(str(n)+" out of "+str(N)+" completed...\n")
dict_sorted = sorted(diction.items(), key=lambda d:d[1], reverse=True)
#filename_total = './output/word_frequency.txt'
#with open(filename_total, 'w') as f:
#    f.write("字\t词频=出现次数*100/总字数")
#    for i in range(len(dict_sorted)):
#        f.write("\n"+dict_sorted[i][0]+"\t"+str(dict_sorted[i][1]))
# Seasons
#filename_seasons = './output/seasons.txt'
#with open(filename_seasons, 'w') as f:
#    f.write("字\t词频=出现次数*100/总字数")
#    f.write("春\t"+str(dict['春'])+"\n夏\t"+str(dict['夏'])+"\n秋\t"+str(dict['秋'])+"\n冬\t"+str(dict['冬']))
# Word of 2 characters
#print("Analyzing word of 2 characters frequency")
#import nltk
#ct_ngrams = list(nltk.bigrams(data))
#data_2gram = list(ct_ngrams)
#ngram_list = set(data_2gram)
#dict_2gram = {}
#n = 0
#N = len(ngram_list)
#for word in ngram_list:
#    n += 1
#    dict_2gram[word]=data_2gram.count(word)*100/N
#    if n % 100 == 1:
#        print(str(n)+" out of "+str(N)+" bigrams completed...\n")
#dict_2gram_sorted = sorted(dict_2gram.items(), key=lambda d:d[1], reverse=True)

#filename_2gram = './output/2grams.txt'
#with open(filename_seasons, 'w') as f:
#    f.write("二字词\t词频=出现次数*100/总字数")
#    for i in range(len(dict_2gram_sorted)):
#        f.write("\n"+"".join(dict_2gram_sorted[i][0])+"\t"+str(dict_2gram_sorted[i][1]))

### word analysis
top_words=[]
for x in dict_sorted:
        top_words.append(x[0])

import networkx as nx
G = nx.Graph()
G.add_nodes_from(top_words[0:200])
used_words = top_words[0:200]
# Initialize edges
for x in used_words:
    for y in used_words:
        if x != y:
            G.add_edge(x, y,weight=0)
N = len(data)
for i in range(0,N//5):
    data_poem = set(data[5*i:5*(i+1)])    ### regularization for data. data type of data_poem == string
    for x in data_poem:
        for y in data_poem:
            if {x,y} <= set(used_words) and len({x,y}) == 2:
                G[x][y]['weight']+=1
    if i % 100 == 1:
        print(str(i)+" out of "+str(N//5)+" weight calculation") # screening using a windows of 5
weighted_centrality = nx.betweenness_centrality(G, k=None, normalized=True, weight='weight', endpoints=False, seed=None)
# algorithm used to calculate shorted-path betweenness centrality performs badly, which is probably because of weighted network.
# alternative way
#cfbc =nx.current_flow _betweenness_centrality(G, normalized=True, weight='weight')
weighted_centrality_sorted = sorted(weighted_centrality.items(), key=lambda d:d[1], reverse=True)
for x in G.nodes():
    G.nodes[x]['centrality']=(weighted_centrality[x]*1000000+np.sum([G[x][y]['weight'] for y in G.nodes() if y != x])/1000)
nodelist=[]
values=[]
nodename={}
for x in weighted_centrality_sorted:
    nodename[x[0]]=x[0]
    nodelist.append(x[0])
    values.append(x[1])
nodelist=nodelist
nodesizes = [(x/np.mean(values))*30 for x in values] # increase contrast between large and small values; for small samples, I cannot see significant patters.
layout=nx.drawing.layout.kamada_kawai_layout(G, dist=None, pos=None, weight='centrality', scale=1, center=None, dim=2)
nx.draw(G,nodelist=nodelist,node_size=nodesizes,width=0.1,style='dashed',labels=nodename)
plt.savefig('network.png')
root = os.getcwd()
# borrow from codeforces_vo.3.py
nodelist=nodelist[1:10]
def outputMatrix(name):
    with open(os.path.join(root,name),"w", encoding="utf-8") as f_out:
        f_csv = csv.writer(f_out)
        f_csv.writerow([""]+nodelist)
        for i in range(len(nodelist)):
            tag = nodelist[i]
            # create a spacer, so I can't write everything in a line
            L = []
            for c in nodelist:
                if c != tag:
                    L.append(G[tag][c]['weight'])
                else:
                    L.append(0)
            f_csv.writerow([tag]+L)
        f_out.close()
    return
outputMatrix('Correlation.csv')

Correlation = pd.read_csv("Correlation.csv", index_col=0)


fig, ax = plt.subplots(figsize=(30,20))

sns.heatmap(Correlation, annot=True, fmt="d", linewidths=.5, cmap='viridis')

plt.savefig("Correlation.png")
settings = {}
settings['n'] = 7                   # dimension of word embeddings
settings['window_size'] = 2         # context window +/- center word
settings['min_count'] = 0           # minimum word count
settings['epochs'] = 5000           # number of training epochs
settings['neg_samp'] = 10           # number of negative words to use during training
settings['learning_rate'] = 0.01    # learning rate
np.random.seed(0)                   # set the seed for reproducibility
w2v=word2vec()
#corpus=corpus
training_data = w2v.generate_training_data(settings, corpus)
w2v.train(trainning)