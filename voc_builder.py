from nltk.tokenize import word_tokenize
import pickle

corpus_file = 'data/corpus.txt'
voc_file = 'data/voc.pkl'

voc = {}
index = 0

with open(corpus_file) as f:
    for l in f:
        for word in word_tokenize(l.lower()):
            if word not in voc:
                voc[word] = index
                index += 1

print('Voc size: ',len(voc))
pickle.dump(voc,open(voc_file,'wb'))
