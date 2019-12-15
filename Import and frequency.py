# Include contribution from Hongfei Yan
# Parallel computation to increase performance
from multiprocessing.dummy import Pool as ThreadPool
import string
import nltk
import re
file = open("Tang_poems_utf_8.txt", encoding='utf8')
text = file.read()
file.close()

# split and create regular poems
pattern = u'卷.[0-9]+'
poems = re.split(pattern, text)[1:-1]
regular_poems = []
regular_title = []
for poem in poems:
    tmp_poem = poem.strip('\n\u3000\u3000◎')
    tmp_poem = tmp_poem.replace('\u3000\u3000', '').split('\n')
    regular_title.append(tmp_poem[0])
    regular_poems.append('\n'.join(tmp_poem[1:]))
# Word frequency
# regular_poems[-1] = regular_poems[-1].replace('（《全唐诗》完）', '')
data = ("".join(regular_poems))

# remove Chinese punctuations
chinese_punc = u'！|？|｡|。|＂|＃|＄|％|＆|＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|\
    ＠|［|＼|］|＾|＿|｀|｛|｜|｝|～|｟|｠|｢|｣|､|、|〃|》|「|」|『|』|【|】|〔|〕|〖|\
    〗|〘|〙|〚|〛|〜|〝|〞|〟|〰|〾|〿|–|—|‘|’|‛|“|”|„|‟|…|‧|﹏||《|□'
chinese_punc_re = re.compile(chinese_punc)
data = chinese_punc_re.sub("", data)

# remove English punctuations
# string.punctuation, '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
exclude = set(string.punctuation)
data = ''.join(ch for ch in data if ch not in exclude)

# remove other non Chinese chars
other_non_cn = u'[0-9a-zA-Z]|\n'
other_non_cn_re = re.compile(other_non_cn)
data0 = other_non_cn_re.sub("", data)
#data0 = data.replace('\n','').replace('-','')

filename = './output/word_all.txt'
with open(filename, 'w', encoding='utf8') as f:
    f.write("Tang_poems_utf_8.txt中包含的所有字")
    f.write("\n"+data0)
f.close()
stopwords = re.compile(
    u'而|何|乎|乃|其|且|若|所|为|焉|以|因|于|与|也|则|者|之|不|自|得|一|来|去|无|可|是|已|此|的|上|中|兮|三')
# data = stopwords.sub('', data)[0:4000]  # sample calculation; reduced size
data1 = stopwords.sub('', data0)
filename = './output/word_all_wo_stopwords.txt'
with open(filename, 'w', encoding='utf8') as f:
    f.write("Tang_poems_utf_8.txt中包含的所有字，不包含21个停用字")
    f.write("\n"+data1)
f.close()

data = data1[:]
data = list(data)
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
filename_total = './output/word_frequency.txt'
with open(filename_total, 'w', encoding='utf8') as f:
    f.write("字\t词频=出现次数*100/总字数")
    for i in range(len(dict_sorted)):
        f.write("\n"+dict_sorted[i][0]+"\t"+str(dict_sorted[i][1]))
f.close()


# Seasons
filename_seasons = './output/seasons.txt'
with open(filename_seasons, 'w', encoding='utf8') as f:
    f.write("字\t词频=出现次数*100/总字数")
    f.write("\n春\t"+str(word_dict['春'])+"\n夏\t"+str(word_dict['夏']) +
            "\n秋\t"+str(word_dict['秋'])+"\n冬\t"+str(word_dict['冬']))
f.close()

# Word of 2 characters
print("Analyzing word of 2 characters frequency")
ct_ngrams = list(nltk.bigrams(data))
data_2gram = ct_ngrams[:]
ngram_list = set(data_2gram)
dict_2gram = {}
N = len(data_2gram)
dict_2gram = {el: 0 for el in ngram_list}


def par_dict_2gram(word):
    dict_2gram[word] += 1/N


pool = ThreadPool()
pool.map(par_dict_2gram, data_2gram)
pool.close()
pool.join()

dict_2gram_sorted = sorted(
    dict_2gram.items(), key=lambda d: d[1], reverse=True)

filename_2gram = './output/2grams.txt'
with open(filename_2gram, 'w', encoding='utf8') as f:
    f.write("二字词\t词频=出现次数*100/总字数")
    for i in range(len(dict_2gram_sorted)):
        f.write(
            "\n"+"".join(dict_2gram_sorted[i][0])+"\t"+str(dict_2gram_sorted[i][1]))
f.close()
