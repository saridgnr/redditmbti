import torch
from torch import nn
from attn import Attn


class EmoModel(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, output_size, n_layers):
        super(EmoModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, bidirectional=True)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.attn = Attn(hidden_size)

    def forward(self, input_text):
        seq_len = len(input_text.data)
        embedded_words = self.embedding(input_text).view(seq_len, 1, -1)
        last_hidden = self.init_hidden()
        rnn_outputs, hidden = self.lstm(embedded_words, last_hidden)
        attn_weights = self.attn(rnn_outputs)
        attn_weights = attn_weights.squeeze(1).view(seq_len, 1)
        rnn_outputs = rnn_outputs.squeeze(1)
        attn_weights = attn_weights.expand(seq_len, self.hidden_size * 2)
        weigthed_outputs = torch.mul(rnn_outputs, attn_weights)
        output = torch.sum(weigthed_outputs, -2)
        output = self.out(output)
        return output

    def init_hidden(self):
        return (torch.zeros(self.n_layers * 2, 1, self.hidden_size).cuda(),
                torch.zeros(self.n_layers * 2, 1, self.hidden_size).cuda())
