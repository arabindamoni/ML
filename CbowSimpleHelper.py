import numpy as np
from nltk.tokenize import word_tokenize
import pickle
corpus_file = 'data/corpus.txt'
voc_file = 'data/voc.pkl'


class CbowSimpleHelper():

    def __init__(self):
        self.tokens = []
        with open(corpus_file) as f:
            for l in f:
                self.tokens.extend(word_tokenize(l.lower()))
        self.voc = pickle.load(open(voc_file,'rb'))

    def get_voc_size(self):
        return len(self.voc)

    def get_data_size(self):
        return len(self.tokens)

    def get_one_hot(self, word):
        vec = np.zeros(len(self.voc))
        vec[self.voc[word]] = 1.0
        return vec

    def get_batch(self, i, batch_size):
        data = self.tokens[i * batch_size: min((i+1)*batch_size,len(self.tokens))]
        X = []
        Y = []
        for word in data:
            if word != '.':
                X.append(self.get_one_hot(word))
                Y.append(self.get_one_hot(word))
        return X,Y

