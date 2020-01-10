#%%
# 五言绝句
# acknowledgement: major part of code comes from:  https://www.jianshu.com/p/e19b96908c69 & https://github.com/ioiogoo/poetry_generator_Keras
'''

@author: ioiogoo

@date: 2018/1/31 19:33

'''
import sys
from multiprocessing.dummy import Pool as ThreadPool
import tensorflow as tf  # **specifically, please use tensorflow 1.x**
from collections import defaultdict
import pandas as pd
import string
import time
import csv
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import tempfile
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
import random
import gensim.models
import os
import keras
import numpy as np
from keras.callbacks import LambdaCallback
from keras.models import Input, Model, load_model
from keras.layers import LSTM, Dropout, Dense, Flatten, Bidirectional, Embedding, GRU
from keras.optimizers import Adam
#%% Preprocessing
file = open("Tang_poems_utf_8.txt")
text = file.read()
file.close()
# split and create regular poems
weight_file = 'poetry_model.h5'
pattern = u'卷.[0-9]+'
poems = re.split(pattern, text)[1:-1]
regular_poems = []
max_len = 6
batch_size = 32
learning_rate = 0.001

# Exclude Non chinese_characters
chinese_punc = u'！|？|｡|。|＂|＃|＄|％|＆|＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|\
    ＠|［|＼|］|＾|＿|｀|｛|｜|｝|～|｟|｠|｢|｣|､|、|〃|》|「|」|『|』|【|】|〔|〕|〖|\
    〗|〘|〙|〚|〛|〜|〝|〞|〟|〰|〾|〿|–|—|‘|’|‛|“|”|„|‟|…|‧|﹏||《|□'

other_non_cn = u'[0-9a-zA-Z]|\n'
exclude = set(string.punctuation)
stopwords = u'而|何|乎|乃|其|且|若|所|为|焉|以|因|于|与|也|则|者|之|不|自|得|一|来|去|无|可|是|已|此|的|上|中|兮|三'
re_auxiliary_words = re.compile("|".join([chinese_punc, other_non_cn]))
regular_sentences = ''
for poem in poems:
    tmp_poem = poem.strip('\n\u3000\u3000◎')
    tmp_poem = tmp_poem.replace('\u3000\u3000', '').split('\n')
    if len(''.join(tmp_poem[1:])) == 24:
        regular_poems.append(''.join(tmp_poem[1:]))
        regular_sentences += "".join(tmp_poem[1:]) + ']'

#%%
# here we use a different way to construct networks in keras, which is specify relations between layers.
w2vmodel = gensim.models.Word2Vec.load('/tmp/w2vmodel')
vocab_list = [word for word, Vocab in w2vmodel.wv.vocab.items()]
word_index = {"": 0}  # initialize "{word: token}" dict
word_vector = {}  # initialize "{word: vector}" dict
embedding_matrix = np.zeros((len(vocab_list) + 1, w2vmodel.vector_size))
predictor = Sequential()
embedding_dim = w2vmodel.vector_size
for i in range(len(vocab_list)):
    word = vocab_list[i]
    word_index[word] = i + 1
    word_vector[word] = w2vmodel.wv[word]
    embedding_matrix[i + 1] = w2vmodel.wv[word]

predictor.add(
    Embedding(len(embedding_matrix),
              embedding_dim,
              weights=[embedding_matrix],
              input_length=max_len))
predictor.add(LSTM(512, return_sequences=True))
predictor.add(Dropout(0.4))
predictor.add(LSTM(256))
predictor.add(Dropout(0.4))
predictor.add(Dense(len(vocab_list), activation='softmax'))

# embedding_dim = w2vmodel.vector_size
# embedd_layer = Embedding(len(embedding_matrix),
#                          embedding_dim,
#                          weights=[embedding_matrix],
#                          input_length=max_len)
# lstm = LSTM(512, return_sequences=True)(
#     embedd_layer
# )  # here we want to aceept a sequence of characters with variable length. we use padding from the left, so it recognizes the data from the hand side
# dropout = Dropout(0.6)(lstm)
# lstm = LSTM(256)(dropout)
# dropout = Dropout(0.6)(lstm)
# dense = Dense(len(vocab_list), activation='softmax')(dropout)
# predictor = Model(
#     inputs=embedd_layer, outputs=dense
# )  # this is our model, we can still build our model with sequence.
predictor.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


def _pred(sentence, temperature=1):
    '''used internally, required sentence long enough, capable of truncating for predictor input'''
    if len(sentence) < max_len:
        print('in def _pred,length error ')
        return

    sentence = sentence[-max_len:]
    x_pred = np.zeros((1, max_len))
    for t, char in enumerate(sentence):
        x_pred[0, t] = word_index[char]
    preds = predictor.predict(
        x_pred, verbose=0
    )[0]  # only one-row, but still one-row; a prediction is a probability distribution here
    next_index = sample(preds, temperature=temperature)
    next_char = vocab_list[next_index]
    return next_char


def _preds(sentence, length=23, temperature=1):
    '''
    sentence: input
    length: length of output
    len(sentence) >= max_len as a requirement
    '''
    sentence = sentence[:max_len]
    generate = ''
    for _ in range(length):
        pred = _pred(sentence, temperature)
        generate += pred
        sentence = sentence[1:] + pred  # it's like a moving reading frame
    return generate


def predict_via_init(char, temperature=1):
    '''
    write a stylistic poem of 5 characters according to the designated initial character
    '''
    if not predictor:
        print('model not loaded')
        return
    index = random.randint(0, len(regular_poems))
    # 选取随机一首诗的最后max_len字符+给出的首个文字作为初始输入
    sentence = regular_poems[index][ - max_len:-1] + char
    generate = str(char)
    # print('first line = ',sentence)
    # 直接预测后面23个字符. 为了简单起见，只有五言绝句被考虑进来
    generate += _preds(sentence, length=23, temperature=temperature)
    return generate


def sample(preds, temperature=1.0):
    '''
    当temperature=1.0时，模型输出正常
    当temperature=0.5时，模型输出比较open
    当temperature=1.5时，模型输出比较保守
    在训练的过程中可以看到temperature不同，结果也不同
    就是一个概率分布变换的问题，保守的时候概率大的值变得更大，选择的可能性也更大
    '''
    preds = np.asarray(preds).astype('float64')
    exp_preds = np.power(preds, 1. / temperature)
    preds = exp_preds / np.sum(exp_preds)
    pro = np.random.choice(range(len(preds)), 1, p=preds)
    return int(pro.squeeze())


def data_generator():
    '''生成器生成数据, 数据比较大，不能直接喂一个大数组进去'''
    i = 0
    while 1:
        x = regular_sentences[i:i + max_len]
        y = regular_sentences[i + max_len]
        # print(x)
        if ']' in x or ']' in y:
            i += 1
            continue
        y_vec = np.zeros(shape=(1, len(vocab_list)), dtype=np.bool)
        if y in vocab_list:
            y_vec[0, word_index[y]] = 1.0  # one-hop encoding
        else:
            i+=1
            continue
        x_vec = np.zeros(shape=(1,max_len))

        for t in range(len(x)):
            if x[t] in vocab_list:
                x_vec[0, t] = word_index[x[t]]
            else:
                x_vec[0, t] = 0

        yield x_vec, y_vec
        i += 1


def predict_random(temperature=1):
    '''随机从库中选取一句开头的诗句，生成五言绝句'''
    init = ''
    while init not in vocab_list:
        index = random.randint(0, len(regular_poems))
        init = regular_poems[index][-2]
    
    generate = predict_via_init(init, temperature=temperature)
    return generate


def generate_sample_result(epoch, logs):
    '''训练过程中，每4个epoch打印出当前的学习情况'''
    if epoch % 4 != 0:
        return

    with open('output/poetry.txt', 'a', encoding='utf-8') as f:
        f.write(
            '==================Epoch {}=====================\n'.format(epoch))

    print("\n==================Epoch {}=====================".format(epoch))
    for diversity in [0.7, 1.0, 1.3]:
        print("------------Diversity {}--------------".format(diversity))
        generate = predict_random(temperature=diversity)
        print(generate)

        # 训练时的预测结果写入txt
        with open('output/poetry.txt', 'a', encoding='utf-8') as f:
            f.write(generate + '\n')


def train():
    '''self_defined program'''
    print('training')
    number_of_epoch = len(regular_sentences) - (max_len +
                                                1) * len(regular_poems)
    number_of_epoch /= batch_size
    number_of_epoch = int(number_of_epoch / 1.5)
    print('epoches = ', number_of_epoch)
    print('poems_num = ', len(regular_poems))
    print('len(content) = ', len(regular_sentences))

    predictor.fit_generator(
        generator=data_generator(),
        verbose=True,
        steps_per_epoch=batch_size,
        epochs=number_of_epoch,
        callbacks=[
            keras.callbacks.ModelCheckpoint(weight_file,
                                            save_weights_only=False),
            LambdaCallback(on_epoch_end=None)
        ])


#%%
train()

# %%
