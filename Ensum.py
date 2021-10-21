import pickle
import numpy as np
import os
import sys
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
#from utils.EvaluationMetrics import compute_bleu
#from utils.EvaluationMetrics import compute_metrics
from utils.IgnoreRate import CalculateIgnoreRate
from utils.VocabularyAndMapping import BuildMappings
from utils.codeerentialEvolution import Population
from utils.Word2Vec import Embedding

calculation_refs = 0


class Calculation_Refs:
    def __init__(self, similarity, train_sums, ignore_rate, mapping, code_list,
                 sum_list1, sum_list2, sum_list3, gen_sums1, gen_sums2, gen_sums3, valid_sums):
        self.similarity = similarity
        self.train_sums = train_sums
        self.ignore_rate = ignore_rate
        self.mapping = mapping
        self.code_list = code_list
        self.sum_list1 = sum_list1
        self.sum_list2 = sum_list2
        self.sum_list3 = sum_list3
        self.gen_sums1 = gen_sums1
        self.gen_sums2 = gen_sums2
        self.gen_sums3 = gen_sums3
        self.valid_sums = valid_sums

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
    Vcb_path = './Vocabulary/cross_data/train.'+type+'.pkl'
    pkl = open(Vcb_path, 'rb')
    vocabulary = pickle.load(pkl)
    return vocabulary

def IsCandidate(sentence1, sentence2):
    a = [x for x in sentence1 if x in sentence2]
    if len(a) < 1:
        return 0
    else:
        return 1

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

def Recall_and_Precision(code, sum, ignoreRate, mapping, alpha):
    total_num = len(code)
    trans_num = 0
    recall = 0
    sum_words = []
    for word in code:
        if ignoreRate[word] < alpha:
            trans_num += IsCandidate(mapping[word], sum)
        else:
            total_num -= 1
        for sum_word in mapping[word]:
            sum_words.append(sum_word)
    correct_num = 0
    for word in sum:
        if word in sum_words:
            correct_num += 1
    precision = 0
    if total_num > 0:
        recall = trans_num / total_num
        if (len(sum)) == 0:
            precision = 0
        else:
            precision = float(correct_num) / float(len(sum))
    return recall, precision

def compute_similarity(train_codes, test_codes):
    counter = CountVectorizer()
    transformer = TfidfTransformer()
    train_matrix = transformer.fit_transform(counter.fit_transform(train_codes))
    test_matrix = transformer.transform(counter.transform(test_codes))

    train_embedding, test_embedding = Embedding(train_codes, test_codes, dataset)

    train_matrix_tfidf = train_matrix
    test_matrix_tfidf = test_matrix

    train_matrix = []
    test_matrix = []

    count = 0
    for embeddings in train_embedding:
        temp = []
        idx = 0
        for embedding in embeddings:
            temp.append(embedding * train_matrix_tfidf[count, idx])
            idx += 1
        temp = np.sum(np.array(temp), axis=0) / len(temp)
        train_matrix.append(temp)
        count += 1
    train_matrix = np.array(train_matrix)

    count = 0
    for embeddings in test_embedding:
        temp = []
        idx = 0
        for embedding in embeddings:
            temp.append(embedding * test_matrix_tfidf[count, idx])
            idx += 1
        temp = np.sum(np.array(temp), axis=0) / len(temp)
        test_matrix.append(temp)
        count += 1
    test_matrix = np.array(test_matrix)

    similarities = cosine_similarity(test_matrix, train_matrix)

    return similarities

def Merge_Generation_File(Output_path, alpha, lambda_f):
    global calculation_refs
    fw = open(Output_path, "w")
    count = -1
    for code in calculation_refs.code_list:
        count += 1
        sum1 = calculation_refs.sum_list1[count]
        sum2 = calculation_refs.sum_list2[count]
        sum3 = calculation_refs.sum_list3[count]
        m1recall, m1precision = Recall_and_Precision(code, sum1, calculation_refs.ignore_rate, calculation_refs.mapping, alpha)
        m2recall, m2precision = Recall_and_Precision(code, sum2, calculation_refs.ignore_rate, calculation_refs.mapping, alpha)
        m3recall, m3precision = Recall_and_Precision(code, sum3, calculation_refs.ignore_rate, calculation_refs.mapping, alpha)
        m1f1score = (1 - lambda_f) * m1precision + lambda_f * m1recall
        m2f1score = (1 - lambda_f) * m2precision + lambda_f * m2recall
        m3f1score = (1 - lambda_f) * m3precision + lambda_f * m3recall

        sum_ref = [calculation_refs.train_sums[calculation_refs.similarity[count].argsort()[-1]].split(' ')]
        smooth = SmoothingFunction()
        bleu1 = sentence_bleu(sum_ref, calculation_refs.gen_sums1[count].split(' '),
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
        bleu2 = sentence_bleu(sum_ref, calculation_refs.gen_sums2[count].split(' '),
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
        bleu3 = sentence_bleu(sum_ref, calculation_refs.gen_sums3[count].split(' '),
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)

        if m1f1score > m2f1score:
            if m1f1score > m3f1score:
                fw.write(calculation_refs.gen_sums1[count] + '\n')
            elif m1f1score < m3f1score:
                fw.write(calculation_refs.gen_sums3[count] + '\n')
            else:
                if bleu1 >= bleu3:
                    fw.write(calculation_refs.gen_sums1[count] + '\n')
                else:
                    fw.write(calculation_refs.gen_sums3[count] + '\n')
        elif m1f1score < m2f1score:
            if m2f1score > m3f1score:
                fw.write(calculation_refs.gen_sums2[count] + '\n')
            elif m2f1score < m3f1score:
                fw.write(calculation_refs.gen_sums3[count] + '\n')
            else:
                if bleu2 >= bleu3:
                    fw.write(calculation_refs.gen_sums2[count] + '\n')
                else:
                    fw.write(calculation_refs.gen_sums3[count] + '\n')
        else:
            if m2f1score > m3f1score:
                if bleu1 >= bleu2:
                    fw.write(calculation_refs.gen_sums1[count] + '\n')
                else:
                    fw.write(calculation_refs.gen_sums2[count] + '\n')
            elif m2f1score < m3f1score:
                fw.write(calculation_refs.gen_sums3[count] + '\n')
            else:
                if bleu1 >= bleu2:
                    if bleu1 >= bleu3:
                        fw.write(calculation_refs.gen_sums1[count] + '\n')
                    else:
                        fw.write(calculation_refs.gen_sums3[count] + '\n')
                else:
                    if bleu2 >= bleu3:
                        fw.write(calculation_refs.gen_sums2[count] + '\n')
                    else:
                        fw.write(calculation_refs.gen_sums3[count] + '\n')
    fw.close()

def Merge(alpha, lambda_f):
    global calculation_refs
    output = []
    count = -1
    for code in calculation_refs.code_list:
        count += 1
        sum1 = calculation_refs.sum_list1[count]
        sum2 = calculation_refs.sum_list2[count]
        sum3 = calculation_refs.sum_list3[count]
        m1recall, m1precision = Recall_and_Precision(code, sum1, calculation_refs.ignore_rate, calculation_refs.mapping, alpha)
        m2recall, m2precision = Recall_and_Precision(code, sum2, calculation_refs.ignore_rate, calculation_refs.mapping, alpha)
        m3recall, m3precision = Recall_and_Precision(code, sum3, calculation_refs.ignore_rate, calculation_refs.mapping, alpha)
        m1f1score = (1 - lambda_f) * m1precision + lambda_f * m1recall
        m2f1score = (1 - lambda_f) * m2precision + lambda_f * m2recall
        m3f1score = (1 - lambda_f) * m3precision + lambda_f * m3recall

        sum_ref = [calculation_refs.train_sums[calculation_refs.similarity[count].argsort()[-1]].split(' ')]

        smooth = SmoothingFunction()
        bleu1 = sentence_bleu(sum_ref, calculation_refs.gen_sums1[count].split(' '),
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
        bleu2 = sentence_bleu(sum_ref, calculation_refs.gen_sums2[count].split(' '),
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
        bleu3 = sentence_bleu(sum_ref, calculation_refs.gen_sums3[count].split(' '),
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)

        if m1f1score > m2f1score:
            if m1f1score > m3f1score:
                output.append(calculation_refs.gen_sums1[count])
            elif m1f1score < m3f1score:
                output.append(calculation_refs.gen_sums3[count])
            else:
                if bleu1 >= bleu3:
                    output.append(calculation_refs.gen_sums1[count])
                else:
                    output.append(calculation_refs.gen_sums3[count])
        elif m1f1score < m2f1score:
            if m2f1score > m3f1score:
                output.append(calculation_refs.gen_sums2[count])
            elif m2f1score < m3f1score:
                output.append(calculation_refs.gen_sums3[count])
            else:
                if bleu2 >= bleu3:
                    output.append(calculation_refs.gen_sums2[count])
                else:
                    output.append(calculation_refs.gen_sums3[count])
        else:
            if m2f1score > m3f1score:
                if bleu1 >= bleu2:
                    output.append(calculation_refs.gen_sums1[count])
                else:
                    output.append(calculation_refs.gen_sums2[count])
            elif m2f1score < m3f1score:
                output.append(calculation_refs.gen_sums3[count])
            else:
                if bleu1 >= bleu2:
                    if bleu1 >= bleu3:
                        output.append(calculation_refs.gen_sums1[count])
                    else:
                        output.append(calculation_refs.gen_sums3[count])
                else:
                    if bleu2 >= bleu3:
                        output.append(calculation_refs.gen_sums2[count])
                    else:
                        output.append(calculation_refs.gen_sums3[count])
    return compute_bleu(output, [calculation_refs.valid_sums], is_file=False)['Bleu_4']

def obj_fuc(v):
    # ret_scores, _, _, rouge = Merge(v[0], v[1])
    # Bleu = ret_scores['Bleu_4']
    # METEOR = ret_scores['METEOR']
    # return -(Bleu+METEOR+rouge)
    Bleu = Merge(v[0], v[1])
    return -Bleu

def EnsGen(dataset, alpha, lambda_f):
    print("Dataset: %s  alpha: %0.2f  lambda: %0.2f" % (dataset, alpha, lambda_f))
    approaches = ["crossrencos", "crossdeep", "crossnmt"]
    train_code_path = "./" + dataset + "/" + "train.txt.src"
    train_sum_path = "./" + dataset + "/" + "train.txt.tgt"
    test_code_path = "./" + dataset + "/" + "test.txt.src"
    gen_sum_path1 = "./" + approaches[0] + "/" + "test.out"
    gen_sum_path2 = "./" + approaches[1] + "/" + "test.out"
    gen_sum_path3 = "./" + approaches[2] + "/" + "test.out"
    EnsGen_output_path = "./archive/" + dataset + "/" + dataset + ".gen.sum"

    Mapping_path = "./Mapping/" + dataset + ".npy"
    if not os.path.exists(Mapping_path):
        BuildMappings(dataset)
    Mapping = load_npy(Mapping_path)
    IgnoreRate_path = "./IgnoreRate/" + dataset + ".npy"
    if not os.path.exists(IgnoreRate_path):
        CalculateIgnoreRate(dataset)
    IgnoreRate = load_npy(IgnoreRate_path)

    code_vocabulary = load_vocabulary(dataset, 'code')
    sum_vocabulary = load_vocabulary(dataset, 'sum')

    train_codes = load_data(train_code_path)
    test_codes = load_data(test_code_path)
    train_sums = load_data(train_sum_path)
    counter1 = CountVectorizer(lowercase=True, vocabulary=code_vocabulary)
    train_code_matrix = counter1.fit_transform(train_codes)
    test_code_matrix = counter1.transform(test_codes)

    similarity = cosine_similarity(test_code_matrix, train_code_matrix)
    # similarity = compute_similarity(train_codes, test_codes)

    gen_sums1 = load_data(gen_sum_path1)
    gen_sums2 = load_data(gen_sum_path2)
    gen_sums3 = load_data(gen_sum_path3)
    counter2 = CountVectorizer(lowercase=True, vocabulary=sum_vocabulary)
    gen_sum_matrix1 = counter2.transform(gen_sums1)
    gen_sum_matrix2 = counter2.transform(gen_sums2)
    gen_sum_matrix3 = counter2.transform(gen_sums3)

    code_nonzero = sparse.csr_matrix(test_code_matrix).nonzero()
    sum_nonzero1 = sparse.csr_matrix(gen_sum_matrix1).nonzero()
    sum_nonzero2 = sparse.csr_matrix(gen_sum_matrix2).nonzero()
    sum_nonzero3 = sparse.csr_matrix(gen_sum_matrix3).nonzero()
    code_list = nonzero2list(code_nonzero)
    sum_list1 = nonzero2list(sum_nonzero1)
    sum_list2 = nonzero2list(sum_nonzero2)
    sum_list3 = nonzero2list(sum_nonzero3)

    test_sum_path = "./" + dataset + "/" + "test.txt.tgt"
    test_sums = load_data(test_sum_path)

    global calculation_refs
    calculation_refs = Calculation_Refs(similarity, train_sums, IgnoreRate, Mapping, code_list,
                                        sum_list1, sum_list2, sum_list3, gen_sums1, gen_sums2, gen_sums3, test_sums)
    Merge_Generation_File(EnsGen_output_path, alpha, lambda_f)
    print("Done. Evaluate the metrics...")

    #compute_metrics(EnsGen_output_path, [test_sum_path])

def Ensum_train(dataset):
    print("Dataset: %s " % (dataset))
    #The three code summarization approaches
    approaches = ["crossrencos", "crossdeep", "crossnmt"]
    train_code_path = "./" + dataset + "/" + "train.txt.src"
    train_sum_path = "./" + dataset + "/" + "train.txt.tgt"
    valid_code_path = "./" + dataset + "/" + "valid.txt.src"
    gen_sum_path1 = "./" + approaches[0] + "/" + "valid.out"
    gen_sum_path2 = "./" + approaches[1] + "/" + "valid.out"
    gen_sum_path3 = "./" + approaches[2] + "/" + "valid.out"

    Mapping_path = "./Mapping/" + dataset + ".npy"
    if not os.path.exists(Mapping_path):
        BuildMappings(dataset)
    Mapping = load_npy(Mapping_path)
    print('mapping over')
    IgnoreRate_path = "./IgnoreRate/" + dataset + ".npy"
    if not os.path.exists(IgnoreRate_path):
        CalculateIgnoreRate(dataset)
    IgnoreRate = load_npy(IgnoreRate_path)
    print('ignore rate over')
    code_vocabulary = load_vocabulary(dataset, 'code')
    sum_vocabulary = load_vocabulary(dataset, 'sum')

    train_codes = load_data(train_code_path)
    valid_codes = load_data(valid_code_path)
    train_sums = load_data(train_sum_path)
    counter1 = CountVectorizer(lowercase=True, vocabulary=code_vocabulary)
    train_code_matrix = counter1.fit_transform(train_codes)
    test_code_matrix = counter1.transform(valid_codes)

    similarity = cosine_similarity(test_code_matrix, train_code_matrix)
    # similarity = compute_similarity(train_codes, valid_codes)

    gen_sums1 = load_data(gen_sum_path1)
    gen_sums2 = load_data(gen_sum_path2)
    gen_sums3 = load_data(gen_sum_path3)
    counter2 = CountVectorizer(lowercase=True, vocabulary=sum_vocabulary)
    gen_sum_matrix1 = counter2.transform(gen_sums1)
    gen_sum_matrix2 = counter2.transform(gen_sums2)
    gen_sum_matrix3 = counter2.transform(gen_sums3)

    code_nonzero = sparse.csr_matrix(test_code_matrix).nonzero()
    sum_nonzero1 = sparse.csr_matrix(gen_sum_matrix1).nonzero()
    sum_nonzero2 = sparse.csr_matrix(gen_sum_matrix2).nonzero()
    sum_nonzero3 = sparse.csr_matrix(gen_sum_matrix3).nonzero()
    code_list = nonzero2list(code_nonzero)
    sum_list1 = nonzero2list(sum_nonzero1)
    sum_list2 = nonzero2list(sum_nonzero2)
    sum_list3 = nonzero2list(sum_nonzero3)
    print('start de')
    valid_sum_path = "./" + dataset + "/" + "valid.txt.tgt"
    valid_sums = load_data(valid_sum_path)
    global calculation_refs
    calculation_refs = Calculation_Refs(similarity, train_sums, IgnoreRate, Mapping, code_list,
                                        sum_list1, sum_list2, sum_list3, gen_sums1, gen_sums2, gen_sums3, valid_sums)

    de = Population(min_range=0, max_range=1, dim=2, factor=0.5, rounds=5, size=10, object_func=obj_fuc)
    _, thresholds = de.evolution()

    EnsGen(dataset, thresholds[0], thresholds[1])

if __name__ == '__main__':
    #The dataset name
    dataset = 'cross_data'
    Ensum_train(dataset)
