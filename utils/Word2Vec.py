import numpy as np
import gensim
import os.path
from sklearn.metrics.pairwise import cosine_similarity

count = 0

def load_data(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')[0:-1]
    lines = [l.strip() for l in lines]
    return lines

def word2vec(dataset):
    train_code_path = "./" + dataset + "/" + "train.txt.src"
    valid_code_path = "./" + dataset + "/"+ "valid.txt.src"
    test_code_path = "./"+ dataset + "/" + "test.txt.src"

    train_code = load_data(train_code_path)
    valid_code = load_data(valid_code_path)
    test_code = load_data(test_code_path)

    sentences = []
    for sentence in train_code:
        sentences.append(np.array(str(sentence).strip().split(' ')).tolist())
    for sentence in valid_code:
        sentences.append(np.array(str(sentence).strip().split(' ')).tolist())
    for sentence in test_code:
        sentences.append(np.array(str(sentence).strip().split(' ')).tolist())
    model = gensim.models.Word2Vec(sentences, sg=1, size=100, window=5, min_count=2, negative=3, sample=0.001, hs=1,
                                   workers=4)

    model.save('./Vocabulary/'+dataset+'.model')
    return model

def embedding_sentences(sentences, model):
    ss = []
    for sentence in sentences:
        s = []
        for word in sentence.split(' '):
            if model.wv.__contains__(word):
                s.append(model.wv.__getitem__(str(word.strip())))
            else:
                s.append(np.zeros(100))
        ss.append(s)
    return ss

def Embedding(train_codes, test_codes, dataset):
    # train_code_path = '.\\data\\'+dataset+'\\'+dataset+'\\.train.code'
    # test_code_path = '.\\data\\'+dataset+'\\'+dataset+'\\.test.code'
    # train_sum_path = '.\\data\\'+dataset+'\\'+dataset+'\\.train.sum'

    # train_code = load_data(train_code_path)
    # test_code = load_data(test_code_path)
    # train_sum = load_data(train_sum_path)
    # code = train_code + test_code

    if os.path.exists('./Vocabulary/'+dataset+'.model'):
        model = gensim.models.Word2Vec.load('./Vocabulary/'+dataset+'.model')
    else:
        model = word2vec(dataset)

    train_code_matrix = embedding_sentences(train_codes, model)
    test_code_matrix = embedding_sentences(test_codes, model)
    # train_sum_matrix = embedding_sentences(train_sum)

    return train_code_matrix, test_code_matrix
