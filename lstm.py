 
# -*- coding: utf-8 -*-
"""
用word2vec和lstm对短文本进行情感分析
"""
import imp
import sys
imp.reload(sys)
import numpy as np
import pandas as pd
import jieba
import re
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from gensim.corpora.dictionary import Dictionary
import multiprocessing
from sklearn.model_selection import train_test_split
import yaml
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
 
vocab_dim = 256
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 5
input_length = 100
 
cpu_count = multiprocessing.cpu_count()
# 加载训练文件
def loadfile():
    neg = pd.read_excel('E:/data/neg.xlsx',header=None,index=None)
    pos = pd.read_excel('E:/data/pos.xlsx',header=None,index=None)
    combined = np.concatenate((np.array(neg[0]).astype(str), np.array(pos[0]).astype(str)))
    y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))
 
    return combined , y
# 获取停用词
def getstopword(stopwordPath):
    stoplist = set()
    for line in stopwordPath:
        stoplist.add(line.strip())
        # print line.strip()
    return stoplist
 
# 分词并剔除停用词
def tokenizer(text):
    stopwordPath = open('E:/data/chineseStopWords.txt','r')
    stoplist = getstopword(stopwordPath)
    stopwordPath.close()
    text_list = []
    for document in text:
 
        seg_list = jieba.cut(document.strip())
        fenci = []
 
        for item in seg_list:
            if item not in stoplist and re.match(r'-?\d+\.?\d*', item) == None and len(item.strip()) > 0:
                fenci.append(item)
        text_list.append(fenci)
    return text_list
#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量
 
        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print('No data provided...')
#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
 
    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined)
    model.train(combined,total_examples = model.corpus_count,epochs = 50)
    model.save('E:/Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined
def get_data(index_dict,word_vectors,combined,y):
 
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print(x_train.shape,y_train.shape)
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test
##定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
 
    print ('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
 
    print ("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch,verbose=1)
 
    print ("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
 
    yaml_string = model.to_yaml()
    with open('E:/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('E:/stm.h5')
    print ('Test score:', score)
#训练模型，并保存
def train():
    print ('Loading Data...')
    combined,y = loadfile()
    print(len(combined), len(y))
    print('Tokenising...')
    combined = tokenizer(combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors,combined=word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
    print(x_train.shape,y_train.shape)
    train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)
 
def input_transform(string):
    words=tokenizer(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('E:/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined
 
def lstm_predict(string):
    print('loading model......')
    with open('lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)
 
    print('loading weights......')
    model.load_weights('E:/lstm.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(string)
    data.reshape(1,-1)
    #print data
    result=model.predict_classes(data)
    if result[0][0]==1:
        print(string,' positive')
    else:
        print(string,' negative')
if __name__=='__main__':
    #train()
    #string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    #string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    #string='酒店的环境非常好，价格也便宜，值得推荐'
    #string='手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'
    #string='我是傻逼'
    #string='你是傻逼'
    string='屏幕较差，拍照也很粗糙。'
    #string='质量不错，是正品 ，安装师傅也很好，才要了83元材料费'
    #string='东西非常不错，安装师傅很负责人，装的也很漂亮，精致，谢谢安装师傅！'
    #lstm_predict(string)
    print('loading model......')
    with open('E:/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('E:/lstm.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
			  
    data = pd.read_excel('E:/data/k.xlsx', header=None, index=None)
  
    for index, row in data.iterrows():
        string = row[0]
        data1 = input_transform(string)
        data1.reshape(1, -1)
        result = model.predict_classes(data1)
        if result[0][0] == 1:
            data.loc[index,-1] = 1
        else:
            data.loc[index,-1] = 0
        print(index, result[0][0])
    data.to_csv('E:/data/k.csv', encoding='gbk')