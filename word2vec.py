import pickle
from gensim.models.keyedvectors import KeyedVectors

print('loading word2vec model...')
wv = KeyedVectors.load_word2vec_format('model/merge_sgns_bigram_char300.txt', binary=False)
with open('model/word2vec.pkl', 'wb') as f:
    pickle.dump(wv, f)
print('save model successfully')

