# file import
import matplotlib.pyplot as plt
import numpy as np
#import string
import re
import networkx as nx
file = open("Tang_poems_utf_8.txt",encoding='utf8')
text = file.read()
file.close()

# split and create regular poems
# This is the font specified. I'm using Ubuntu, so I downloaded a free Chinese font to allow for matplotlib showing Chinese characters.
#plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']

filename = './output/word_frequency.txt'
file = open(filename, encoding='utf8')
text = file.read()
file.close()

text = text.split('\n')
text = text[1:]

words_df = {}
for item in text:
    word, fq = item.split()
    words_df[word] = float(fq)
    
#dict_sorted = sorted(words_df.items(), key=lambda d: d[1], reverse=True)
dict_sorted = list(words_df.keys())
        
# Given a dictionary, evaluate the co-occurence relationship between top 148 most frequently used words
#used_words = [x[0] for x in dict_sorted[0:147]]
used_words = dict_sorted[0:74]
        
G = nx.Graph()
G.nodes()
for x in used_words:
    for y in set(used_words)-set([x]):
        G.add_edge(x, y, weight=0) if y != x else None
count = 0


filename = './output/regular_poems.txt'
file = open(filename, encoding='utf8')
text = file.read()
file.close()
regular_poems = text.split("\n====\n")

stopwords = re.compile(u'而|何|乎|乃|其|且|若|所|为|焉|以|因|于|与|也|则|者|之|不|\
                       自|得|一|来|去|无|可|是|已|此|的|上|中|兮|三')

import random
#regular_poems = regular_poems[:500]
regular_poems = random.sample(regular_poems, 800)


for x in regular_poems:
    y = stopwords.sub('', x)
    count += 1
    tmp_set = set(list(y))
    for w in used_words:
        for u in set(used_words)-set([w]):
            if set([u, w]).issubset(tmp_set):
                G[w][u]['weight'] = G[w][u]['weight'] + 1
centrality = nx.betweenness_centrality(G, k=None, normalized=True,
                                       weight='weight', endpoints=True, seed=None)
weighted_centrality_sorted = sorted(
    centrality.items(), key=lambda x: x[1], reverse=True)
nodelist = [x[0] for x in weighted_centrality_sorted]
print(weighted_centrality_sorted)
nodelabels = {}
count = 0
for x in nodelist:
    count += 1
    #nodelabels[x] = x if count < 20 else ''
    nodelabels[x] = x 
    
weight_list = [x[1] for x in weighted_centrality_sorted]
#size_list = [((x/np.mean(weight_list))*20)**2 for x in weight_list]

#size_list = [((x/np.mean(weight_list))*20)**2 for x in weight_list]

miu = np.mean(weight_list)
sigma = np.std(weight_list,ddof=0)
size_list = [800*(x-miu)/sigma for x in weight_list]


centrality_layout = nx.spring_layout(
    G, pos=None, weight='weight', scale=1, center=None, dim=2)

#plt.figure(figsize = (20, 20)) 
plt.figure(figsize=(12, 8),dpi=200)

node_color = [float(v) for v in weight_list]

nx.draw(G, pos=centrality_layout, labels=nodelabels, nodelist=nodelist,
        node_size=size_list, 
        linewidth=0.1, font_color='w', edge_color='k',
        font_family='SimHei', node_color=node_color)

plt.axis('off')
#nx.draw(G, font_family='SimHei', with_labels=True,node_size=size_list)
plt.savefig('./output/network.png')
#plt.show()