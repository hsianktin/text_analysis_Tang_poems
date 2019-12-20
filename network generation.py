# file import
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt
import nltk
import numpy as np
import re
import networkx as nx
import string
file = open("Tang_poems_utf_8.txt", encoding='utf8')
text = file.read()
file.close()

# split and create regular poems
# This is the font specified. I'm using Ubuntu, so I downloaded a free Chinese font to allow for matplotlib showing Chinese characters.
# substitute ‘Simsum’ for 'Taipei Sans TC Beta' if your matplotlib has inbuilt chinese font simsum
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
pattern = u'卷.[0-9]+'
poems = re.split(pattern, text)[1:-1]
regular_poems = []
regular_title = []

# Exclude Non chinese_characters
chinese_punc = u'！|？|｡|。|＂|＃|＄|％|＆|＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|\
    ＠|［|＼|］|＾|＿|｀|｛|｜|｝|～|｟|｠|｢|｣|､|、|〃|》|「|」|『|』|【|】|〔|〕|〖|\
    〗|〘|〙|〚|〛|〜|〝|〞|〟|〰|〾|〿|–|—|‘|’|‛|“|”|„|‟|…|‧|﹏||《|□|-'
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
# Word frequency
data = "".join(regular_poems)
data = list(re_auxiliary_words.sub('', data))
wordlist = set(data)
word_dict = {el: 0 for el in wordlist}
n = 0
N = len(data)


def par_dict_word(word):
    word_dict[word] += 1/N


print("Analyzing character frequency")
pool = ThreadPool()
pool.map(par_dict_word, data)
pool.close()
pool.join()
dict_sorted = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
print('Finished sorting')
# Given a dictionary, evaluate the co-occurence relationship between top 148 most frequently used words
used_words = [x[0] for x in dict_sorted[0:200] if x != '一']
G = nx.Graph()
G.nodes()
for x in used_words:
    for y in set(used_words)-set([x]):
        G.add_edge(x, y, weight=0) if y != x else None
count = 0


def par_poem_networks(poem):
    reg_poem = re_auxiliary_words.sub('', poem)
    tmp_set = set(list(reg_poem))
    for w in used_words:
        for u in set(used_words)-set([w]):
            if set([u, w]).issubset(tmp_set):
                G[w][u]['weight'] = G[w][u]['weight']


print("Analyzing between centrality")
pool = ThreadPool()
pool.map(par_poem_networks, regular_poems)
pool.close()
pool.join()
print('Finished')  # parallel computing here doesn't accelerate much

for x in regular_poems:
    y = re_auxiliary_words.sub('', x)
    count += 1
    tmp_set = set(list(y))
    for w in used_words:
        for u in set(used_words)-set([w]):
            if set([u, w]).issubset(tmp_set):
                G[w][u]['weight'] = G[w][u]['weight'] + 1


centrality = nx.betweenness_centrality(G, k=None, normalized=True,
                                       weight='weight', endpoints=True, seed=None)
for x in G.nodes():
    G.nodes[x]['weight'] = centrality[x]
for x, y in G.edges():
    G[x][y]['central'] = G.nodes[x]['weight'] + \
        G.nodes[y]['weight'] + G[x][y]['weight']/100000
weighted_centrality_sorted = sorted(
    centrality.items(), key=lambda x: x[1], reverse=True)
nodelist = [x[0] for x in weighted_centrality_sorted]
nodelabels = {}

weight_list = [x[1] for x in weighted_centrality_sorted]

size_list = [((x/np.mean(weight_list)))*100 for x in weight_list]
count = 0
for i in range(len(size_list)):
    x = nodelist[i]
    nodelabels[x] = x if size_list[i] > 20 else ''
centrality_layout = nx.kamada_kawai_layout(
    G, pos=None, weight='central', scale=5, center=None, dim=2)

nx.draw_networkx(G, pos=centrality_layout, labels=nodelabels, nodelist=nodelist, style='dashed',
                 node_size=size_list, linewidth=0.01, font_color='w', edge_color='y', font_size=8)
plt.savefig('./output/network.png')

# Alternative way to plot the network
