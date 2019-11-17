# coding: utf-8
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
import torch.nn.utils.rnn as rnn
import matplotlib.pyplot as plt
from BiLSTM_ATT import BiLSTM_ATT

train_data_file = 'data/train_data.pkl'
model_file = 'model/model.pkl'

with open(train_data_file, 'rb') as f:
    config = pickle.load(f)
    embeddings = pickle.load(f)
    postags = pickle.load(f)
    positions1 = pickle.load(f)
    positions2 = pickle.load(f)
    targets = pickle.load(f)

num_instance = config['NUM_INSTANCE']
batch_size = 64
sent_len = 120
embeddings, lengths = rnn.pad_packed_sequence(
    embeddings, batch_first=True, total_length=sent_len)
postags, _ = rnn.pad_packed_sequence(
    postags, batch_first=True, total_length=sent_len)
positions1, _ = rnn.pad_packed_sequence(
    positions1, batch_first=True, total_length=sent_len)
positions2, _ = rnn.pad_packed_sequence(
    positions2, batch_first=True, total_length=sent_len)

train_num = int(num_instance * 0.6)
validate_num = int(num_instance * 0.2)
test_num = num_instance - train_num - validate_num

validate = embeddings[train_num:train_num + validate_num]
postags_v = postags[train_num:train_num + validate_num]
positions1_v = positions1[train_num:train_num + validate_num]
positions2_v = positions2[train_num:train_num + validate_num]
targets_v = targets[train_num:train_num + validate_num]
lengths_v = lengths[train_num:train_num + validate_num]

test = embeddings[train_num + validate_num:]
postags_t = postags[train_num + validate_num:]
positions1_t = positions1[train_num + validate_num:]
positions2_t = positions2[train_num + validate_num:]
targets_t = targets[train_num + validate_num:]
lengths_t = lengths[train_num + validate_num:]

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
print('validate set size:', validate_num)
print('test set size:', test_num)

def train_model(model, learning_rate, epoch_num):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,    weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    validate_loss = []
    for epoch in range(epoch_num):
        print('epoch=', epoch)

        # train
        for sent, postag, pos1, pos2, target, length in dataloader:
            y = model(sent, postag, pos1, pos2, length)
            loss = criterion(y, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())

        # validation
        y_v = model(validate, postags_v, positions1_v, positions2_v, lengths_v)
        loss_v = criterion(y_v, targets_v)
        validate_loss.append(loss_v.item())
    print('finished')
    
    return train_loss, validate_loss


def ave(inp):
    out = []
    for i in range(1, len(inp) + 1):
        out.append(np.average(inp[:i]))
    return out

def loss_graphic(train_loss, validate_loss, title):
    plt.plot(train_loss, 'r-', label='train')
    plt.plot(validate_loss, 'b-', label='validate')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title(title)
    plt.show()


def test_model(model, test, postags_t, posi1_t, posi2_t, lengths_t, targets_t):
    y_t = model(test, postags_t, posi1_t, posi2_t, lengths_t)
    acc = (y_t.max(dim=1)[1] == targets_t).sum().item() / test_num
    recall = torch.mul(y_t.max(dim=1)[1], targets_t).sum(
    ).item() / targets_t.sum().item()
    return (acc, recall)

model_config = {}
model_config['SENTENCE_SIZE'] = sent_len
model_config['EMBED_SIZE'] = config['EMBEDDING_LENGTH']
model_config['RELATION_SIZE'] = config['RELATION_LENGTH']
model_config['POSTAG_SIZE'] = config['POS_LENGTH']
model_config['POSTAG_DIM'] = 20
model_config['POSITION_SIZE'] = 120
model_config['POSITION_DIM'] = 50
model_config['HIDDEN_DIM'] = 200
model = BiLSTM_ATT(model_config)

train_loss, valid_loss = train_model(model, 0.005, 150)
loss_graphic(ave(train_loss), ave(valid_loss), 'epoch=150 hidden=200')
print(valid_loss[-1])

acc, recall = test_model(model, test, postags_t, positions1_t, positions2_t, lengths_t, targets_t)
print('accuracy:', acc)
print('recall:', recall)

# save
print('saving model...')
with open(model_file, 'wb') as f:
    torch.save(model, f)
