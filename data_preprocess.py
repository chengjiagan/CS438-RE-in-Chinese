# coding: utf-8
import pickle
import json
import numpy as np
import pyltp
import torch
from torch.nn.utils.rnn import pad_sequence
from gensim import corpora

corups_file = 'data/corups.txt'
word2vec_file = 'model/word2vec.wv'
dict_file = 'data/dictionary.dict'
cws_model = 'model/cws.model'
pos_model = 'model/pos.model'
ner_model = 'model/ner.model'
save = 'data/train_data_small.pkl'

# one-hot embedding of POS tag
id2pos = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm',
          'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz',
          'o', 'p', 'q', 'r', 'u', 'v', 'wp', 'ws', 'x', 'z']
pos2id = {}
for i, p in enumerate(id2pos):
    pos2id[p] = i

# load the dictionary
dictionary = corpora.Dictionary.load(dict_file)

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
        if n[0] == 'O':
            words.append(w)
            pos.append(p)
        elif n[0] == 'S':
            words.append(w)
            pos.append(n[-2:].lower())
        elif n[0] == 'B':
            ne = [w]
            pos.append(n[-2:].lower())
        elif n[0] == 'I':
            ne.append(w)
        elif n[0] == 'E':
            ne.append(w)
            words.append(''.join(ne))
    return (words, pos)

# process samples from dataset
# words_samples = []  # for debug
idx_samples = []
postag_samples = []
position1_samples = []
position2_samples = []
length_samples = []
label_samples = []
max_length = 250
num_pos = 0
num_neg = 0


with open(corups_file, encoding='utf-8') as f:
    for line in f:
        json_obj = json.loads(line)
        words, pos = segment(json_obj['text'])
        length = len(words)
        if length > max_length:
            continue
        words_idx = torch.tensor(dictionary.doc2idx(words)) + 1 # 1 is the offset
        pos_ids = torch.tensor([pos2id[p] for p in pos])

        for nh, ni, l in zip(json_obj['person'], json_obj['org'], json_obj['label']):
            nh_index = words.index(nh)
            ni_index = words.index(ni)
            nh_position = list(range(-nh_index, length - nh_index))
            ni_position = list(range(-ni_index, length - ni_index))
            nh_position = torch.tensor(nh_position)
            ni_position = torch.tensor(ni_position)

            # words_samples.append(words)
            idx_samples.append(words_idx)
            postag_samples.append(pos_ids)
            position1_samples.append(nh_position)
            position2_samples.append(ni_position)
            length_samples.append(length)
            if l:
                num_pos += 1
                label_samples.append(1)
            else:
                num_neg += 1
                label_samples.append(0)

print('positive instances:', num_pos)
print('negative instances:', num_neg)

# shuffle the samples
pack_samples = list(zip(idx_samples, postag_samples, position1_samples, position2_samples, label_samples, length_samples))
np.random.shuffle(pack_samples)
idx_samples, postag_samples, position1_samples, position2_samples, label_samples, length_samples = zip(*pack_samples)

# pad the sequence using pytorch's helper function torch.nn.utils.rnn.pad_sequence()
idx_samples = pad_sequence(idx_samples, batch_first=True)
postag_samples = pad_sequence(postag_samples, batch_first=True)
position1_samples = pad_sequence(position1_samples, batch_first=True)
position2_samples = pad_sequence(position2_samples, batch_first=True)
length_samples = torch.tensor(length_samples)
label_samples = torch.tensor(label_samples)

# save
config = {}
config['RELATION_LENGTH'] = 2  # len(id2relation)
config['POS_LENGTH'] = len(id2pos)
config['NUM_INSTANCE'] = num_pos + num_neg
config['NUM_POS_INSTANCE'] = num_pos
config['NUM_NEG_INSTANCE'] = num_neg
config['MAX_LENGTH'] = max_length
print('saving...')
with open(save, "wb") as f:
    pickle.dump(config, f)
    pickle.dump(idx_samples, f)
    pickle.dump(postag_samples, f)
    pickle.dump(position1_samples, f)
    pickle.dump(position2_samples, f)
    pickle.dump(label_samples, f)
    pickle.dump(length_samples, f)
