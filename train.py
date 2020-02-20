# coding: utf-8
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
import matplotlib.pyplot as plt
from BiLSTM_ATT import BiLSTM_ATT

train_data_file = 'data/train_data_small.pkl'
wordvec_file = 'model/embedding.npy'
model_file = 'model/model.pkl'

with open(train_data_file, 'rb') as f:
    config = pickle.load(f)
    idx_samples = pickle.load(f)
    postag_samples = pickle.load(f)
    position1_samples = pickle.load(f)
    position2_samples = pickle.load(f)
    label_samples = pickle.load(f)
    length_samples = pickle.load(f)

num = config['NUM_INSTANCE']
train_num = int(num * 0.8)
validate_num = int(num * 0.2)

train = idx_samples[:train_num]
postag_t = postag_samples[:train_num]
positions1_t = position1_samples[:train_num]
positions2_t = position2_samples[:train_num]
label_t = label_samples[:train_num]
lengths_t = length_samples[:train_num]
dataset = D.TensorDataset(train, postag_t, positions1_t,
                          positions2_t, label_t, lengths_t)

validate = idx_samples[train_num:]
postags_v = postag_samples[train_num:]
positions1_v = position1_samples[train_num:]
positions2_v = position2_samples[train_num:]
label_v = label_samples[train_num:]
lengths_v = length_samples[train_num:]

print('train set size:', train.size(0))
print('validate set size:', validate.size(0))

def get_metric(y_t, targets_t):
    prec = torch.mul(y_t.max(dim=1)[1], targets_t).sum().item() / y_t.max(dim=1)[1].sum().item()
    recall = torch.mul(y_t.max(dim=1)[1], targets_t).sum().item() / targets_t.sum().item()
    f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1

def train_model(model, dataloader, epoch_num):
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    validate_loss = []
    validate_metric = []
    for epoch in range(epoch_num):
        if epoch % 10 == 0:
            print('epoch:', epoch)

        # train
        model.train()
        for sent, postag, pos1, pos2, target, length in dataloader:
            y = model(sent, postag, pos1, pos2, length)
            loss = criterion(y, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())

        # validation
        model.eval()
        y_v = model(validate, postags_v, positions1_v, positions2_v, lengths_v)
        loss_v = criterion(y_v, label_v)
        validate_loss.append(loss_v.item())
        validate_metric.append(get_metric(y_v, label_v))
    print('finished')

    return train_loss, validate_loss, validate_metric


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

def metric_graphic(valid_metric, title):
    acc, rec, f1 = zip(*valid_metric)
    plt.plot(acc, 'r-', label='accuracy')
    plt.plot(rec, 'b-', label='recall')
    plt.plot(f1, 'g-', labels='f1')
    plt.xlabel('epochs')
    plt.ylabel('metric')
    plt.legend()
    plt.title(title)
    plt.show()

epoch_num = 150
batch_size = 256

model_config = {}
wordvec = np.load(wordvec_file)
model_config['EMBED_SIZE'] = wordvec.shape[1]
model_config['RELATION_SIZE'] = config['RELATION_LENGTH']
model_config['POSTAG_SIZE'] = config['POS_LENGTH']
model_config['POSTAG_DIM'] = 20
model_config['POSITION_SIZE'] = config['MAX_LENGTH']
model_config['POSITION_DIM'] = 50
model_config['HIDDEN_DIM'] = 200
model_config['PRETRAINED_WORDVEC'] = torch.tensor(wordvec, dtype=torch.float)
model = BiLSTM_ATT(model_config)

dataloader = D.DataLoader(dataset, batch_size=batch_size,
                          shuffle=False, num_workers=2, drop_last=True)
train_loss, valid_loss, valid_metric = train_model(model, dataloader, epoch_num)
loss_graphic(ave(train_loss), ave(valid_loss),
             'epoch=%d hidden=%d batch=%d' % (epoch_num, model_config['HIDDEN_DIM'], batch_size))
metric_graphic(valid_metric,
             'epoch=%d hidden=%d batch=%d' % (epoch_num, model_config['HIDDEN_DIM'], batch_size))
acc, recall, f1 = get_metric(model, validate, postags_v,
                         positions1_v, positions2_v, lengths_v, label_v)
print('accuracy:', acc)
print('recall:', recall)
print('f1:', f1)

# save
print('saving model...')
with open(model_file, 'wb') as f:
    torch.save(model, f)
