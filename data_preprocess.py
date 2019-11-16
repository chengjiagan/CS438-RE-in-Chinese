# coding: utf-8
import pickle
import re

import numpy as np
import pyltp
import thulac
import torch
from torch.nn.utils.rnn import pack_sequence

corups_file = ['data/LabeledData.{}.txt'.format(i) for i in range(1, 6)]
word2vec_file = 'model/word2vec.pkl'
cws_model = 'model/cws.model'
pos_model = 'model/pos.model'
ner_model = 'model/ner.model'
save = 'data/train_data.pkl'

# one-hot embedding of POS tag
id2pos = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm',
          'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz',
          'o', 'p', 'q', 'r', 'u', 'v', 'wp', 'ws', 'x', 'z']
pos2id = {}
for i, p in enumerate(id2pos):
    pos2id[p] = i

# four types of relations
# id2relation = ['I', 'B', 'E', 'P']
# relation2id = {}
# for i, r in enumerate(id2relation):
#     relation2id[r] = i

# get the word2vec embedding of a word
print('loading pretrained word2vec model...')
with open(word2vec_file, 'rb') as f:
    word2vec = pickle.load(f)
print('finish loading')
unknown = np.ones(word2vec.vector_size, dtype=np.float32)
def embedding(word):
    if word in word2vec:
        return word2vec[word]
    else:
        em = np.zeros(word2vec.vector_size, dtype=np.float32)
        for c in word:
            if c in word2vec:
                em = em + word2vec[c]
            else:
                em = em + unknown
        return em

# segment words in the sentence with the help of given named entity
segmentor = pyltp.Segmentor()
segmentor.load(cws_model)
postagger = pyltp.Postagger()
postagger.load(pos_model)
recognizer = pyltp.NamedEntityRecognizer()
recognizer.load(ner_model)
def segment(sent):
    words = []
    pos = []
    segments = segmentor.segment(sent)
    postags = postagger.postag(segments)
    netags = recognizer.recognize(segments, postags)
    ne = []
    for w, p, n in zip(segments, postags, netags):
        if n[0] == '0':
            words.append(w)
            pos.append(pos2id[p])
        elif n[0] == 'S':
            words.append(w)
            pos.append(pos2id[n[-2:].lower()])
        elif n[0] == 'B':
            ne = [w]
            pos.append(pos2id[n[-2:].lower()])
        elif n[0] == 'I':
            ne.append(w)
        elif n[0] == 'E':
            ne.append(w)
            words.append(''.join(ne))
    return (words, pos)

# process samples from dataset
words_samples = []  # for debug
embedding_samples = []
postag_samples = []
position1_samples = []
position2_samples = []
labels = []
max_length = 120
num_pos = 0
num_neg = 0
for filename in corups_file:
    print('processing "{}"'.format(filename))

    with open(filename, encoding='utf-8') as f:
        new_sample = True
        sents = []
        for line in f.readlines():
            line = line.strip()

            # end of the sample
            if line == '':
                new_sample = True
                continue

            # new sample
            if new_sample:
                line = re.sub(r'{([^}]*)/n.}', lambda x: x.group(1), line)
                sents = pyltp.SentenceSplitter.split(line)
                new_sample = False
                continue

            # relation labels
            relation = line.split('|')
            # some entity may have several names
            # and some marked entity may not appear in the sentence
            for sent in sents:
                words, postags = segment(sent)
                ne1s = relation[0].split(',')
                ne2s = relation[1].split(',')
                for ne1 in ne1s:
                    if ne1 in words:
                        postags[words.index(ne1)] = pos2id['nh']
                for ne2 in ne2s:
                    if ne2 in words:
                        postags[words.index(ne2)] = pos2id['ni']
                nhs = []
                nis = []
                for i, (w, p) in enumerate(zip(words, postags)):
                    if p == pos2id['nh']:
                        nhs.append((i, w))
                    elif p == pos2id['ni']:
                        nis.append((i, w))

                embeddings = torch.tensor([embedding(w) for w in words])
                postags = torch.tensor(postags)
                for i1, nh in nhs:
                    for i2, ni in nis:
                        positions1 = torch.abs(
                            torch.tensor(range(-i1, len(words) - i1)))
                        positions2 = torch.abs(
                            torch.tensor(range(-i2, len(words) - i2)))
                        if nh in ne1s and ni in ne2s:
                            words_samples.append(words)
                            embedding_samples.append(embeddings)
                            postag_samples.append(postags)
                            position1_samples.append(positions1)
                            position2_samples.append(positions2)
                            labels.append(1)
                            num_pos = num_pos + 1
                        elif num_pos > num_neg:
                            words_samples.append(words)
                            embedding_samples.append(embeddings)
                            postag_samples.append(postags)
                            position1_samples.append(positions1)
                            position2_samples.append(positions2)
                            labels.append(0)
                            num_neg = num_neg + 1
print('positive instances:', num_pos)
print('negative instances:', num_neg)

# pack the variable length sequence using pytorch's pack_sequence
embedding_samples = pack_sequence(embedding_samples, enforce_sorted=False)
postag_samples = pack_sequence(postag_samples, enforce_sorted=False)
position1_samples = pack_sequence(position1_samples, enforce_sorted=False)
position2_samples = pack_sequence(position2_samples, enforce_sorted=False)
labels = torch.tensor(labels)

# save
config = {}
config['EMBEDDING_LENGTH'] = word2vec.vector_size
config['RELATION_LENGTH'] = 2 # len(id2relation)
config['POS_LENGTH'] = len(id2pos)
config['NUM_INSTANCE'] = num_pos + num_neg
print('saving...')
with open(save, "wb") as f:
    pickle.dump(config, f)
    pickle.dump(embedding_samples, f)
    pickle.dump(postag_samples, f)
    pickle.dump(position1_samples, f)
    pickle.dump(position2_samples, f)
    pickle.dump(labels, f)
