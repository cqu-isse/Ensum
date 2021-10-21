import numpy as np
import os
import pickle
# import re
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class obj:
    def __init__(self):
        self.key = 0
        self.weight = 0.0

# load lines from a file
def load_data(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')[0:-1]
    lines = [l.strip() for l in lines]
    return lines

# Caculate CFScore
def Caculate_CFScore(Train_code_Path, Train_sum_Path,dataset):
    train_codes = load_data(Train_code_Path)
    train_sums = load_data(Train_sum_Path)

    # regex = r'[\da-fA-F]{7,}\.{2}[\da-fA-F]{7,}|[\da-fA-F]{30,}'
    # print(re.findall(regex," ".join(train_codes)))

    counter1 = CountVectorizer(lowercase=True)
    code_matrix = counter1.fit_transform(train_codes)
    code_len = len(counter1.vocabulary_)
    counter2 = CountVectorizer(lowercase=True)
    sum_matrix = counter2.fit_transform(train_sums)
    if not os.path.exists("./Vocabulary"):
        os.mkdir("./Vocabulary")
    pkl_name = str(Train_code_Path).split("/")[-1].split(".")
    pkl_code_path = './Vocabulary/' +dataset+'/'+ pkl_name[0] + '.code.pkl'
    print(pkl_code_path)
    with open(pkl_code_path, 'wb') as fw1:
        pickle.dump(counter1.vocabulary_, fw1)
        print('write vocab')
    pkl_sum_path = './Vocabulary/' + dataset+'/'+pkl_name[0] + '.sum.pkl'
    with open(pkl_sum_path, 'wb') as fw2:
        pickle.dump(counter2.vocabulary_, fw2)

    sum_len = len(counter2.vocabulary_)
    similarities = cosine_similarity(code_matrix.T, sum_matrix.T)
    nonzero = sparse.csr_matrix(similarities).nonzero()
    #p rint("calculate similarities done")
    return similarities, nonzero, code_len

def sort_CFlist(similarities, nonzero, code_len, Mapping_output_Path, n:int=10):
    k = 0
    lis = []
    gather = []
    p = -1
    for i in nonzero[0]:
        p = p + 1
        if k == i:
            a = obj()
            a.key = nonzero[1][p]
            a.weight = similarities[i, nonzero[1][p]]
            lis.append(a)
        else:
            lis.sort(key=lambda obj: obj.weight, reverse=True)
            # print(lis)
            gather.append(lis)
            while k < i - 1:
                k = k + 1
                lis = []
                a = obj()
                a.key = -1
                a.weight = -1
                lis.append(a)
                gather.append(lis)
            k = i
            lis = []
            a = obj()
            a.key = nonzero[1][p]
            a.weight = similarities[i, nonzero[1][p]]
            lis.append(a)
    lis.sort(key=lambda obj: obj.weight, reverse=True)
    gather.append(lis)

    # print("sort similarities done")

    nparray = np.zeros([code_len, 10])

    si = -1
    sj = -1
    for i in gather:
        si = si + 1
        for j in i:
            sj = sj + 1
            nparray[si][sj] = j.key

            if sj >= n - 1:
                break
        while sj < n - 1:
            sj = sj + 1
            nparray[si][sj] = -1
        sj = -1
    np.save(Mapping_output_Path, nparray)
    print("save Mappings: done")

def BuildMappings(dataset):
    if not os.path.exists("./Mapping"):
        # print("build Mapping dir: done")
        os.mkdir("./Mapping")
    Train_code_Path = "./" + dataset + "/" + "train.txt.src"
    Train_sum_Path = "./" + dataset + "/" + "train.txt.tgt"
    Mapping_output_Path = "./Mapping/" + dataset + ".npy"
    similarities, nonzero, code_len = Caculate_CFScore(Train_code_Path, Train_sum_Path,dataset)
    sort_CFlist(similarities, nonzero,code_len, Mapping_output_Path)
