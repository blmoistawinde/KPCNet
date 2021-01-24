import torch
import torch.nn as nn
from torch.nn import functional as F
from constants import *
from hparams import hparams
import copy

class GRUKwdPredictor(nn.Module):
    def __init__(self, word_embeddings, hidden_size, kwd_size,
                 gru_layers=2, mlp_layers=1, dropout=0.2):
        super(GRUKwdPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.out_size = kwd_size
        self.gru_layers = gru_layers
        self.mlp_layers = mlp_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(len(word_embeddings), len(word_embeddings[0]))
        self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.embedding.weight.requires_grad = hparams.UPDATE_WD_EMB
        self.gru = nn.GRU(len(word_embeddings[0]), hidden_size, gru_layers, dropout=self.dropout, bidirectional=True)
        self.mlp = nn.Sequential(*[nn.Linear(hidden_size, hidden_size) for i in range(mlp_layers)])
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, kwd_size)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        outputs, hidden = self.gru(embedded, hidden)
        input_states = torch.mean(hidden, dim=0)
        x = self.mlp(input_states)
        logits = self.proj(self.dropout(x))
        
        return logits

class CNNKwdPredictor(nn.Module):
    def __init__(self, word_embeddings, kwd_size, mlp_layers=1,
                 filter_num=100, filter_sizes=(3,4,5), dropout=0.2):
        super(CNNKwdPredictor, self).__init__()

        chanel_num = 1
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.hidden_size = len(filter_sizes) * filter_num
        self.mlp_layers = mlp_layers
        self.embedding = nn.Embedding(len(word_embeddings), len(word_embeddings[0]))
        self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.embedding.weight.requires_grad = hparams.UPDATE_WD_EMB
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, len(word_embeddings[0]))) for size in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(*[nn.Linear(self.hidden_size, self.hidden_size) for i in range(mlp_layers)])
        self.fc = nn.Linear(self.hidden_size, kwd_size)

    def forward(self, input_seqs, input_lengths, hidden=None):
        x = self.embedding(input_seqs.transpose(0,1))
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

class CNNPostKwdPredictor(nn.Module):
    def __init__(self, word_embeddings, kwd_size, mlp_layers=1,
                 filter_num=100, filter_sizes=(3,4,5), dropout=0.2):
        super(CNNPostKwdPredictor, self).__init__()

        chanel_num = 1
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.hidden_size = len(filter_sizes) * filter_num
        self.mlp_layers = mlp_layers
        self.embedding = nn.Embedding(len(word_embeddings), len(word_embeddings[0]))
        self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.embedding.weight.requires_grad = hparams.UPDATE_WD_EMB
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, len(word_embeddings[0]))) for size in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(*[nn.Linear(self.hidden_size, self.hidden_size) for i in range(mlp_layers)])
        self.fc1 = nn.Linear(self.hidden_size, kwd_size)
        self.fc2 = nn.Linear(kwd_size, kwd_size)    # model the inter-relation of keywords

    def forward(self, input_seqs, input_lengths, hidden=None):
        x = self.embedding(input_seqs.transpose(0,1))
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        self.logits_pre = self.fc1(x)
        residual = self.fc2(F.relu(self.logits_pre))
        logits = self.logits_pre + residual
        return logits

def get_predictor(word_embeddings, hparams):
    assert hparams.KWD_PREDICTOR_TYPE in {"gru", "cnn", "cnnpost"}
    if hparams.KWD_PREDICTOR_TYPE == "gru":
        kwd_predictor = GRUKwdPredictor(word_embeddings, hparams.HIDDEN_SIZE, hparams.MAX_KWD, hparams.RNN_LAYERS,
                                        hparams.KWD_MODEL_LAYERS, hparams.DROPOUT)
    elif hparams.KWD_PREDICTOR_TYPE == "cnn":
        kwd_predictor = CNNKwdPredictor(word_embeddings, hparams.MAX_KWD,
                                        hparams.KWD_MODEL_LAYERS, dropout=hparams.DROPOUT)
    elif hparams.KWD_PREDICTOR_TYPE == "cnnpost":
        kwd_predictor = CNNPostKwdPredictor(word_embeddings, hparams.MAX_KWD,
                                        hparams.KWD_MODEL_LAYERS, dropout=hparams.DROPOUT)
    return kwd_predictor