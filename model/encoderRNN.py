import torch
import torch.nn as nn
from torch.nn import functional as F
from constants import *
from hparams import hparams
import copy


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, word_embeddings, n_layers=1, dropout=0.1, update_wd_emb=False):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(len(word_embeddings), len(word_embeddings[0]))
        self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.embedding.weight.requires_grad = update_wd_emb
        self.gru = nn.GRU(len(word_embeddings[0]), hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        #packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        #outputs, hidden = self.gru(packed, hidden)
        #outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        # outputs of shape `(seq_len, batch, num_directions * hidden_size)`
        # hidden of shape `(num_layers * num_directions, batch, hidden_size)`
        outputs, hidden = self.gru(embedded, hidden)
        # (seq_len, batch, hidden_size)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden

