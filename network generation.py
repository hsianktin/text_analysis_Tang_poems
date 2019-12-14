# file import
import matplotlib.pyplot as plt
import nltk
import re
import networkx as nx
file = open("Tang_poems_utf_8.txt")
text = file.read()
file.close()

# split and create regular poems
# This is the font specified. I'm using Ubuntu, so I downloaded a free Chinese font to allow for matplotlib showing Chinese characters.
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
pattern = u'卷.[0-9]+'
poems = re.split(pattern, text)[1:1000]
regular_poems = []
regular_title = []
for poem in poems:
    tmp_poem = poem.strip('\n\u3000\u3000◎')
    tmp_poem = tmp_poem.replace('\u3000\u3000', '').split('\n')
    regular_title.append(tmp_poem[0])
    regular_poems.append('\n'.join(tmp_poem[1:]))
# Word frequency
data = ("".join(regular_poems)).replace('\n', "").replace('。', '').replace('，', '').replace(
    '：', '').replace('；', '').replace('？', '').replace('！', '')
stopwords = re.compile(
    '而|何|乎|乃|其|且|然|若|所|为|焉|也|以|矣|于|之|则|者|与|欤|因|[\u002d|\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]|[0-9]|[a-zA-Z]')  # stopwords correction Chinese marks -> unicode string
# only w2v calculation need truncation; but it still requires time to run
data = stopwords.sub('', data)
data = list(data)
wordlist = set(data)
dict = {}
n = 0
N = len(data)
for word in wordlist:
    n += 1
    dict[word] = data.count(word)*100/N
    print(str(n)+" out of "+str(len(wordlist)) +
          " is completed...\n") if n % 100 == 0 else None
dict_sorted = sorted(dict.items(), key=lambda d: d[1], reverse=True)
filename_total = './output/word_frequency.txt'
with open(filename_total, 'w') as f:
    f.write("字\t词频=出现次数*100/总字数")
    for i in range(len(dict_sorted)):
        f.write("\n"+dict_sorted[i][0]+"\t"+str(dict_sorted[i][1]))
# Given a dictionary, evaluate the co-occurence relationship between top 148 most frequently used words
used_words = [x[0] for x in dict_sorted[0:147]]
G = nx.Graph()
G.nodes()
for x in used_words:
    for y in used_words:
        G.add_edge(x, y, weight=0) if y != x else None

for x in regular_poems:
    y = x.split('\n')
    for z in y:
        for w in used_words:
            if
