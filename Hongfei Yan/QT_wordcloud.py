

from wordcloud import WordCloud
import matplotlib.pylab as plt

font = "C:\Windows\Fonts\STXINGKA.TTF"
wc = WordCloud(background_color="white",    #   背景颜色
               max_words=300,               #   最大词数
               #mask=back_color,            #   掩膜，产生词云背景的区域，以该参数值作图绘制词云，这个参数不为空时，width,height会被忽略
               max_font_size=80,            #   显示字体的最大值
               #stopwords=STOPWORDS.add("差评"),   #   使用内置的屏蔽词，再添加一个
               font_path=font,              #   解决显示口字型乱码问题，可进入C:/Windows/Fonts/目录更换字体
               random_state=42,             #   为每一词返回一个PIL颜色
               prefer_horizontal=10)        #   调整词云中字体水平和垂直的多少


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


wc.fit_words(words_df)

plt.figure(figsize = (10, 10)) 
plt.imshow(wc)
plt.axis("off")
plt.tight_layout(pad = 0) 
  
plt.show() 
