import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn

class BiLSTM_ATT(nn.Module):
    def __init__(self, config):
        super(BiLSTM_ATT, self).__init__()

        self.embed_dim = config['EMBED_SIZE']
        self.relation_size = config['RELATION_SIZE']
        self.postag_size = config['POSTAG_SIZE']
        self.postag_dim = config['POSTAG_DIM']
        self.position_size = config['POSITION_SIZE']
        self.position_dim = config['POSITION_DIM']
        self.hidden_dim = config['HIDDEN_DIM']

        self.word_embed = nn.Embedding.from_pretrained(config['PRETRAINED_WORDVEC'], freeze=False)
        self.postag_embed = nn.Embedding(self.postag_size, self.postag_dim)
        self.position1_embed = nn.Embedding(
            self.position_size,
            self.position_dim)
        self.position2_embed = nn.Embedding(
            self.position_size,
            self.position_dim)

        self.input_size = self.embed_dim + self.postag_dim
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True)

        self.att_layer = nn.Linear(
            self.embed_dim + self.position_dim * 2,
            1, bias=False)

        self.hidden2relation = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.hidden_dim * 4, self.relation_size)
        )

    def attention(self, tensor):
        return F.softmax(self.att_layer(torch.tanh(tensor)), dim=1)

    def forward(self, sent, postag, pos1, pos2, length):
        sent = self.word_embed(sent)
        postag = self.postag_embed(postag)
        pos1 = self.position1_embed(torch.abs(pos1))
        pos2 = self.position2_embed(torch.abs(pos2))
        inp = torch.cat((sent, postag), 2)
        inp = rnn.pack_padded_sequence(
            inp, length, batch_first=True, enforce_sorted=False)

        hidden, _ = self.lstm(inp)

        hidden, _ = rnn.pad_packed_sequence(hidden, batch_first=True, total_length=sent.size(1))
        att = torch.cat((sent, pos1, pos2), 2)
        att = self.attention(att).transpose(1, 2)
        hidden = torch.bmm(att, hidden)

        output = self.hidden2relation(hidden).view(-1, self.relation_size)

        return F.softmax(output, dim=1)
