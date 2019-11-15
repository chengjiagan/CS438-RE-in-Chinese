import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F


class BiLSTM_ATT(nn.Module):
    def __init__(self, config):
        super(BiLSTM_ATT, self).__init__()

        self.sent_size = config['SENTENCE_SIZE']
        self.embed_dim = config['EMBED_SIZE']
        self.relation_size = config['RELATION_SIZE']
        self.postag_size = config['POSTAG_SIZE']
        self.postag_dim = config['POSTAG_DIM']
        self.position_size = config['POSITION_SIZE']
        self.position_dim = config['POSITION_DIM']
        self.hidden_dim = config['HIDDEN_DIM']

        self.postag_embed = nn.Embedding(self.postag_size, self.postag_dim)
        self.position1_embed = nn.Embedding(
            self.position_size,
            self.position_dim)
        self.position2_embed = nn.Embedding(
            self.position_size,
            self.position_dim)

        # self.input_size = self.embed_dim + self.pos_dim + self.position_dim * 2
        self.lstm = nn.LSTM(
            input_size=self.embed_dim + self.postag_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bidirectional=True)

        self.att_layer = nn.Linear(
            self.hidden_dim * 2 + self.position_dim * 2,
            1)

        self.hidden2relation = nn.Linear(
            self.hidden_dim * 2, self.relation_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.hidden_dim),
                torch.zeros(2, batch_size, self.hidden_dim))

    def attention(self, tensor):
        return F.softmax(self.att_layer(torch.tanh(tensor)), dim=1)

    def forward(self, sent, postag, pos1, pos2, length):
        batch_size = sent.size(0)

        postag = self.postag_embed(postag)
        embed = torch.cat((sent, postag), 2)
        embed = rnn.pack_padded_sequence(
            embed, length, batch_first=True, enforce_sorted=False)

        hidden = self.init_hidden(batch_size)
        hidden, _ = self.lstm(embed, hidden)

        hidden, _ = rnn.pad_packed_sequence(
            hidden, batch_first=True, total_length=self.sent_size)
        pos1 = self.position1_embed(pos1)
        pos2 = self.position2_embed(pos2)
        att = torch.cat((hidden, pos1, pos2), 2)
        att = self.attention(att).transpose(1, 2)
        hidden = torch.bmm(att, hidden)

        output = self.hidden2relation(hidden).view(-1, self.relation_size)

        return F.softmax(output, dim=1)
