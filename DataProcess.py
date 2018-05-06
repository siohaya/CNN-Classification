import numpy as np
import pickle 
from collections import defaultdict
import re
import pandas as pd
import os.path


def load_bin_vec(fname, vocab):
    """
    Loads 200x1 word vecs from 日本語 Wikipedia エンティティベクトル (東北大学) word2vec
    """
    word_vecs = {}
    with open(fname, "r") as f:
        header = f.readline() #读取vec的第一行
        #print(header)
        #print(header.split())
        vocab_size, layer1_size = map(int, header.split())
        for X in range(vocab_size):
        #word = []
            line=f.readline()
            l = re.split("\s+", line,  1)
        #print(l[0])
        #print(vocab_size,layer1_size)
        #word.append(l[0])
        #print(word)
            if l[0] in vocab:
                word_vecs[l[0]] = np.fromstring(l[1], dtype=np.float64,sep=' ')
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=200):
    """
    For words not in word2vec
    Give a random vector

    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)


def get_W(word_vecs, k=200):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs) #出现在推特的 字典的长度
    word_idx_map = dict() 
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_train(filename='train.csv'):
    train = pd.read_csv(filename, header=None)
    X, y = train[1], train[0]
    X = X.tolist()
    return X, y


def load_test(filename='test.csv'):
    train = pd.read_csv(filename, header=None)
    X, y = train[1], train[0]
    X = X.tolist()
    return X, y


def build_data_cv(clean_string=True):
    """
    Loads data.
    """
    revs = []
    vocab = defaultdict(float) #defaultdict类就好像是一个dict，但是它是使用一个类型来初始化的：defaultdict类的初始化函数接受一个类型作为参数，当所访问的键不存在的时候，可以实例化一个值作为默认值：
    train_x, train_y = load_train()
    test_x, test_y = load_test()
    for i in range(len(train_x)):
        rev = []#定义一个新列表
        rev.append(train_x[i].strip())#append将第i个推特(.strip移除头尾空格 )插入rev列表中
        if clean_string:
            orig_rev = clean_str(" ".join(rev))#将第i个推特前面加个空格 然后进行清理
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split()) #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集
                                      #split() 根据空格对第i个推特进行切割 然后丢进 words这个set里面
        for word in words:
            vocab[word] += 1  #将存在于words中的所有单词进行编号
        datum = {"y":train_y[i],
                  "text": orig_rev,
                  "num_words": len(orig_rev.split()),
                  "split": 1
                }
        #print(vocab)
#y标识 text清理完的推特前后无空格 中间所有东西有空格 num_words 当前推特有多少单词
        revs.append(datum)
        # revs存有所有datum train的split标识为1 test的split的标识为0
    for i in range(len(test_x)):
        rev = []
        rev.append(test_x[i].strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split()) #words里面是乱序的
        for word in words:
            vocab[word] += 1 
        datum = {"y":test_y[i],
                  "text": orig_rev,
                  "num_words": len(orig_rev.split()),
                  "split": 0
                }
        revs.append(datum)

    return revs, vocab #vocab里面是所有单词 以及初始化了1.0


def clean_str(string, TREC=False):
    re.compile('(http|https):\/\/[^\/"]+[^ |"]*')
    string = re.sub('(http|https):\/\/[^\/"]+[^ |"]*', "http", string) #将地址换成 http
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def process_data(file_name):

    print ("creating dataset...")

    # load data
    print ("loading data...")
    revs, vocab = build_data_cv(clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    #print ()"data loaded!"
    print ("number of sentences: " + str(len(revs)))
    print ("vocab size: " + str(len(vocab)))
    print ("max sentence length: " + str(max_l))

    # load word2vec
    print ("loading word2vec vectors...")
    w2v_file = 'entity_vector.model.txt'
    w2v = load_bin_vec(w2v_file, vocab)
    print ("num words already in word2vec: " + str(len(w2v)))
    print ("word2vec loaded!")

    #Addind random vectors for all unknown words
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)

    # dump to pickle file
    pickle.dump([revs, W, W2, word_idx_map, vocab, max_l], open(file_name, "wb"))

    print ("over!")

process_data("twitter.p")