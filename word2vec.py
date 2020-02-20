# coding: utf-8
import torch
import json
import pyltp
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim import corpora

cws_model = 'model/cws.model'
pos_model = 'model/pos.model'
ner_model = 'model/ner.model'
segmentor = pyltp.Segmentor()
segmentor.load(cws_model)
postagger = pyltp.Postagger()
postagger.load(pos_model)
recognizer = pyltp.NamedEntityRecognizer()
recognizer.load(ner_model)
def segment(sent):
    words = []
    # pos = []
    segments = segmentor.segment(sent)
    postags = postagger.postag(segments)
    netags = recognizer.recognize(segments, postags)
    ne = []
    for w, _, n in zip(segments, postags, netags):
        if n[0] == 'O':
            words.append(w)
            # pos.append(p)
        elif n[0] == 'S':
            words.append(w)
            # pos.append(n[-2:].lower())
        elif n[0] == 'B':
            ne = [w]
            # pos.append(n[-2:].lower())
        # elif n[0] == 'I':
            ne.append(w)
        elif n[0] == 'E':
            ne.append(w)
            words.append(''.join(ne))
    return words

print('loading word2vec model...')
wv = KeyedVectors.load('model/word2vec.wv')
print('finished')

corups = []
with open('data/corups.txt', encoding='utf-8') as f:
    for line in f:
        json_obj = json.loads(line, encoding='utf-8')
        corups.append([w for w in segment(json_obj['text']) if w in wv])

dictionary = corpora.Dictionary(corups)
embedding = np.vstack(([np.zeros(300)], [wv[dictionary[i]] for i in range(len(dictionary))]))

np.save('model/embedding.npy', embedding)
dictionary.save('data/dictionary.dict')
