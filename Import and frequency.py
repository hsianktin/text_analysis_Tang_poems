# file import
import nltk
import re
file = open("Tang_poems_utf_8.txt")
text = file.read()
file.close()

# split and create regular poems
pattern = u'卷.[0-9]+'
poems = re.split(pattern, text)[1:]
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
    u'而|何|乎|乃|其|且|然|若|所|为|焉|也|以|矣|于|之|则|者|与|欤|因|(|)|[1-9]|-')
data = stopwords.sub('', data)  # only w2v calculation need truncation
data = list(data)
wordlist = set(data)
dict = {}
n = 0
N = len(wordlist)
for word in wordlist:
    n += 1
    dict[word] = data.count(word)*100/N
    print(str(n)+" out of "+str(N)+" ic completed...\n")
dict_sorted = sorted(dict.items(), key=lambda d: d[1], reverse=True)
filename_total = './output/word_frequency.txt'
with open(filename_total, 'w') as f:
    f.write("字\t词频=出现次数*100/总字数")
    for i in range(len(dict_sorted)):
        f.write("\n"+dict_sorted[i][0]+"\t"+str(dict_sorted[i][1]))
# Seasons
filename_seasons = './output/seasons.txt'
with open(filename_seasons, 'w') as f:
    f.write("字\t词频=出现次数*100/总字数")
    f.write("春\t"+str(dict['春'])+"\n夏\t"+str(dict['夏']) +
            "\n秋\t"+str(dict['秋'])+"\n冬\t"+str(dict['冬']))
# Word of 2 characters
print("Analyzing word of 2 characters frequency")
ct_ngrams = list(nltk.bigrams(data))
data_2gram = list(ct_ngrams)
ngram_list = set(data_2gram)
dict_2gram = {}
n = 0
N = len(ngram_list)
for word in ngram_list:
    n += 1
    dict_2gram[word] = data_2gram.count(word)*100/N
    print(str(n)+" out of "+str(N)+" ic completed...\n") if n % 10 == 0
dict_2gram_sorted = sorted(
    dict_2gram.items(), key=lambda d: d[1], reverse=True)

filename_2gram = './output/2grams.txt'
with open(filename_seasons, 'w') as f:
    f.write("二字词\t词频=出现次数*100/总字数")
    for i in range(len(dict_2gram_sorted)):
        f.write(
            "\n"+"".join(dict_2gram_sorted[i][0])+"\t"+str(dict_2gram_sorted[i][1]))
