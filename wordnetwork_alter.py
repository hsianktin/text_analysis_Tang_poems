# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


# %% [markdown]
#
# # Create a network with color map
#
#
# Draw a graph with matplotlib, color edges.
# You must have matplotlib>=87.7 for this to work.
#

# %%
# file import
import time
import sys
import plotly
import plotly.graph_objects as go
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
pattern = u'卷.[0-9]+'
poems = re.split(pattern, text)[1:-1]
regular_poems = []
regular_title = []

# Exclude Non chinese_characters
chinese_punc = u'！|？|｡|。|＂|＃|＄|％|＆|＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|    ＠|［|＼|］|＾|＿|｀|｛|｜|｝|～|｟|｠|｢|｣|､|、|〃|》|「|」|『|』|【|】|〔|〕|〖|    〗|〘|〙|〚|〛|〜|〝|〞|〟|〰|〾|〿|–|—|‘|’|‛|“|”|„|‟|…|‧|﹏||《|□|-'
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
# %%
# Given a dictionary, evaluate the co-occurence relationship between top 148 most frequently used words
used_words = [x[0] for x in dict_sorted[0:150]]
bigram_used_words = [set([x, y])
                     for x in used_words for y in used_words if x != y]
G = nx.Graph()
G.nodes()
for x in used_words:
    for y in set(used_words)-set([x]):
        G.add_edge(x, y, weight=0) if y != x else None
count = 0


def par_poem_networks(poem):
    cpunc = u'。|，'
    sentences = re.split(cpunc, poem)
    for tmp_set in sentences:
        tmp_set = set(tmp_set)
        for s in bigram_used_words:
            if s.issubset(tmp_set):
                G[list(s)[0]][list(s)[1]]['weight'] += 1


print("Analyzing between centrality")
# There is no visible acceleration based on multithreading. Maybe the memory is locked by manipulating the same object
pool = ThreadPool(8)
cnt = 0
for _ in pool.imap(par_poem_networks, regular_poems):
    sys.stdout.write('done %d/%d\r' % (cnt, len(regular_poems)))
    cnt += 1
pool.close()
pool.join()

print('Finished')  # parallel computing here doesn't accelerate much


centrality = nx.betweenness_centrality(G, k=None, normalized=True,
                                       weight='weight', endpoints=True, seed=None)

for x in G.nodes():
    G.nodes[x]['weight'] = centrality[x]
    G.nodes[x]['rec_wei'] = 1/centrality[x]

# %% Debugging section
# for x in G.edges(data=True):
#     print(x)


# %%
weighted_centrality_sorted = sorted(
    centrality.items(), key=lambda x: x[1], reverse=True)
nodelist = [x[0] for x in weighted_centrality_sorted]


# weight_list = [x[1] for x in weighted_centrality_sorted]
# nodecolorlist = [x/max(weight_list)*255 for x in weight_list]
# size_list = [((x/np.mean(weight_list)))*50 for x in weight_list]
# print(centrality)
# centrality_layout = nx.kamada_kawai_layout(
#     G, pos=None, weight='weight', scale=5, center=None, dim=2)


# %%
centrality_layout = nx.spring_layout(
    G, k=15, pos=None, weight='weight', scale=5, center=None, dim=2)
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = centrality_layout[edge[0]]
    x1, y1 = centrality_layout[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
for node in G.nodes():
    x, y = centrality_layout[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Between Centrality',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))


node_text = []
node_bc = []
for node in G.nodes():
    node_bc.append(centrality[node])
    node_text.append('Between Centrality of ' +
                     node + ' : '+str(centrality[node]))

node_trace.marker.color = node_bc
node_trace.text = node_text


fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                title='<br>Between-centrality of Top 150 most-frequently-used words',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="Hover to view node text",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002)],
                xaxis=dict(showgrid=False, zeroline=False,
                           showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )


plotly.offline.plot(
    fig, filename='150_word_network_kamda_layout_weight_by_sentence.html')


# %%
print('Fig has been saved as word_network_kamda_layout.html.')
