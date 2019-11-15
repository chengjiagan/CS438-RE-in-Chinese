# coding: utf-8
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
import torch.nn.utils.rnn as rnn
from BiLSTM_ATT import BiLSTM_ATT

train_data_file = 'data/train_data.pkl'
model_file = 'model/model.pkl'

with open(train_data_file, 'rb') as f:
    config = pickle.load(f)
    id2relation = pickle.load(f)
    embeddings = pickle.load(f)
    postags = pickle.load(f)
    positions1 = pickle.load(f)
    positions2 = pickle.load(f)
    targets = pickle.load(f)

num_instance = config['NUM_INSTANCE']
embed_len = config['EMBEDDING_LENGTH']
relation_len = config['RELATION_LENGTH']
pos_len = config['POS_LENGTH']

batch_size = 128
epoch_num = 100
learning_rate = 0.0005
sent_len = 120

embeddings, lengths = rnn.pad_packed_sequence(
    embeddings, batch_first=True, total_length=sent_len)
postags, _ = rnn.pad_packed_sequence(
    postags, batch_first=True, total_length=sent_len)
positions1, _ = rnn.pad_packed_sequence(
    positions1, batch_first=True, total_length=sent_len)
positions2, _ = rnn.pad_packed_sequence(
    positions2, batch_first=True, total_length=sent_len)

train_num = int(num_instance * 0.8)
test_num = num_instance - train_num
test = embeddings[train_num:]
postags_t = postags[train_num:]
positions1_t = positions1[train_num:]
positions2_t = positions2[train_num:]
targets_t = targets[train_num:]
lengths_t = lengths[train_num:]

train = embeddings[:train_num]
postags = postags[:train_num]
positions1 = positions1[:train_num]
positions2 = positions2[:train_num]
targets = targets[:train_num]
lengths = lengths[:train_num]
dataset = D.TensorDataset(train, postags, positions1,
                          positions2, targets, lengths)
dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=False)

print('train set size:', train_num)
print('test set size:', test_num)

model_config = {}
model_config['SENTENCE_SIZE'] = sent_len
model_config['EMBED_SIZE'] = embed_len
model_config['RELATION_SIZE'] = relation_len
model_config['POSTAG_SIZE'] = pos_len
model_config['POSTAG_DIM'] = 50
model_config['POSITION_SIZE'] = 120
model_config['POSITION_DIM'] = 60
model_config['HIDDEN_DIM'] = 300
model = BiLSTM_ATT(model_config)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# train
for epoch in range(epoch_num):
    print('epoch:', epoch)

    # train
    for sent, postag, pos1, pos2, target, length in dataloader:
        y = model(sent, postag, pos1, pos2, length)
        loss = criterion(y, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('train loss:', loss.item())

    # test
    y_t = model(test, postags_t, positions1_t, positions2_t, lengths_t)
    loss_t = criterion(y_t, targets_t)
    print('test loss:', loss_t.item())

y_t = model(test, postags_t, positions1_t, positions2_t, lengths_t)

# save
print('saving model...')
with open(model_file, 'wb') as f:
    torch.save(model, f)
