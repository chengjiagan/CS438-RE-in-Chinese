# coding: utf-8
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
from torch.autograd import Variable

train_data_file = 'data/train_data.pkl'

with open(train_data_file, 'rb') as f:
    config = pickle.load(f)
    id2relation = pickle.load(f)
    embeddings = pickle.load(f)
    postags = pickle.load(f)
    positions1 = pickle.load(f)
    positions2 = pickle.load(f)
    labels = pickle.load(f)

num_instance = config['NUM_INSTANCE']
embed_len    = config['EMBEDDING_LENGTH']
relation_len = config['RELATION_LENGTH']
pos_len      = config['POS_LENGTH']
sent_len     = config['SENTENCE_LENGTH']

batch_size = 120
epichs = 100
learning_rate = 0.0005

train_num = int(num_instance * 0.8)
test_num = num_instance - train_num
test = torch.tensor(embeddings[train_num:train_num + test_num])
postags_t = torch.tensor(postags[train_num:train_num + test_num])
positions1_t = torch.tensor(positions1[train_num:train_num + test_num])
positions2_t = torch.tensor(positions2[train_num:train_num + test_num])
labels_t = torch.tensor(labels[train_num:train_num + test_num])
dataset_t = D.TensorDataset(test, postags_t, positions1_t, positions2_t, labels_t)
dataloader_t = D.DataLoader(dataset_t, batch_size=batch_size, shuffle=True, num_workers=2)

train = torch.tensor(embeddings[:train_num])
postags = torch.tensor(postags[:train_num])
positions1 = torch.tensor(positions1[:train_num])
positions2 = torch.tensor(positions2[:train_num])
labels = torch.tensor(labels[:train_num])
dataset = D.TensorDataset(train, postags, positions1, positions2, labels)
dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
