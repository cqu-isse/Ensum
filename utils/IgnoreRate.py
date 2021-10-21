import os
import pickle
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from .ybVocabularyAndMapping import BuildMappings

# load lines from a file
def load_data(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')[0:-1]
    lines = [l.strip() for l in lines]
    return lines

# load a numpy.array from a .npy file
def load_npy(npypath):
    return np.load(npypath)

def load_vocabulary(dataset, type):
    Vcb_path =  './Vocabulary/'+dataset+'/train.'+type+'.pkl'
    pkl = open(Vcb_path, 'rb')
    vocabulary = pickle.load(pkl)
    return vocabulary

# judge the word mapped or not
def IsCandidate(sentence1, sentence2):
    a = [x for x in sentence1 if x in sentence2]
    if len(a)<1:
        return 1
    else:
        return 0

def nonzero2list(nonzero):
    k = 0
    lis = []
    gather = []
    p = -1
    for i in nonzero[0]:
        p = p + 1
        if k == i:
            lis.append(nonzero[1][p])
        else:
            gather.append(lis)
            while k < i - 1:
                k = k + 1
                lis = []
                gather.append(lis)
            lis = []
            k = i
            lis.append(nonzero[1][p])
    gather.append(lis)
    return gather

def CalculateIgnoreRate(dataset):
    code_path = "./" + dataset + "/" + "train.txt.src"
    sum_path = "./" + dataset + "/" + "train.txt.tgt"
    codes = load_data(code_path)
    sums = load_data(sum_path)

    Mapping_input_Path = "./Mapping/" + dataset + ".npy"
    if not os.path.exists(Mapping_input_Path):
        BuildMappings(dataset)
    Mapping = load_npy(Mapping_input_Path)

    code_vocabulary = load_vocabulary(dataset, 'code')
    sum_vocabulary = load_vocabulary(dataset, 'sum')

    counter1 = CountVectorizer(lowercase=True, vocabulary=code_vocabulary)
    code_matrix = counter1.fit_transform(codes)
    code_len = len(counter1.vocabulary_)
    counter2 = CountVectorizer(lowercase=True, vocabulary=sum_vocabulary)
    sum_matrix = counter2.fit_transform(sums)

    code_nonzero = sparse.csr_matrix(code_matrix).nonzero()
    sum_nonzero = sparse.csr_matrix(sum_matrix).nonzero()
    code_list = nonzero2list(code_nonzero)
    sum_list = nonzero2list(sum_nonzero)

    rate_array = np.zeros([code_len,2])
    count = -1
    for code in code_list:
        count = count + 1
        sum = sum_list[count]
        for word in code:
            rate_array[word][0] += IsCandidate(Mapping[word],sum)
            rate_array[word][1] += 1
    count_word_num = -1
    for rate in rate_array:
        count_word_num += 1
        rate[0] = rate[0] / rate[1]

    rate_array_Path = "./IgnoreRate/" + dataset + ".npy"
    if not os.path.exists("./IgnoreRate"):
        os.mkdir("./IgnoreRate")
        # print("build IgnoreRate dir: done")
    np.save(rate_array_Path, [x[0] for x in rate_array])
    print("save IgnoreRate: done")
