import torch
import torch.nn as nn
from torch.nn import functional as F
from constants import *
from hparams import hparams
import copy

class MLPBridge(nn.Module):
    def __init__(self, hidden_size, kwd_size, ebridge_dim, dbridge_dim,
                mlp_layers=1, ebridge_layer=1, dbridge_layer=1, norm_type='dropout', dropout=0.2):
        super(MLPBridge, self).__init__()

        self.hidden_size = hidden_size
        self.kwd_size = kwd_size
        self.ebridge_dim = ebridge_dim
        self.dbridge_dim = dbridge_dim
        self.mlp_layers = mlp_layers
        self.ebridge_layers = ebridge_layer
        self.dbridge_layer = dbridge_layer
        self.norm_type = norm_type

        if self.norm_type == 'dropout':
            self.norm_layer = nn.Dropout(dropout)
            self.dropout = dropout
        elif self.norm_type == 'sigmoid':
            self.norm_layer = nn.Sigmoid()
        elif self.norm_type == 'batch_norm':
            self.norm_layer = nn.BatchNorm1d(kwd_size)
        elif self.norm_type == 'layer_norm':
            self.norm_layer = nn.LayerNorm(kwd_size)
        else:
            self.norm_layer = None
        self.mlp = nn.Sequential(*([nn.Linear(kwd_size, hidden_size)]+
                                    [nn.Linear(hidden_size, hidden_size) for i in range(mlp_layers-1)]))
        self.encoder_bridge = nn.Sequential(*([nn.Linear(hidden_size, ebridge_dim)] +
                                              [nn.Linear(ebridge_dim, ebridge_dim) for i in range(ebridge_layer - 1)]))
        self.decoder_bridge = nn.Sequential(*([nn.Linear(hidden_size, dbridge_dim)] +
                                              [nn.Linear(dbridge_dim, dbridge_dim) for i in range(dbridge_layer - 1)]))

    def forward(self, logits, kwd_mask=None):
        if self.norm_type != 'none':
            x = self.norm_layer(logits)
        else:
            x = logits
        if kwd_mask is None:
            feature = self.mlp(x)
        elif hparams.HARD_KWD_BRIDGE:
            feature = self.mlp(kwd_mask)
        else:
            # use sampling or provided
            feature = self.mlp(x * kwd_mask)

        e_feature = self.encoder_bridge(feature)
        d_feature = self.decoder_bridge(feature)
        return e_feature, d_feature

# deprecated
class MemoryBridge(nn.Module):
    def __init__(self, word_embeddings, kwd_size, ebridge_dim, dbridge_dim, mlp_layers=1, 
                 ebridge_layer=1, dbridge_layer=1, memory_hops=2):
        super(MemoryBridge, self).__init__()

        self.kwd_size = kwd_size
        self.ebridge_dim = ebridge_dim
        self.dbridge_dim = dbridge_dim
        self.mlp_layers = mlp_layers
        self.ebridge_layers = ebridge_layer
        self.dbridge_layer = dbridge_layer
        self.memory_hops = memory_hops

        self.wd_emb_dim = len(word_embeddings[0])
        self.embedding = nn.Embedding(len(word_embeddings), self.wd_emb_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.embedding.weight.requires_grad = hparams.UPDATE_WD_EMB


        self.mlp = nn.Sequential(*([nn.Linear(kwd_size, self.wd_emb_dim)] +
                                    [nn.Linear(self.wd_emb_dim, self.wd_emb_dim) for i in range(mlp_layers-1)]))
        self.encoder_bridge = nn.Sequential(*([nn.Linear(self.wd_emb_dim, ebridge_dim)] +
                                              [nn.Linear(ebridge_dim, ebridge_dim) for i in range(ebridge_layer - 1)]))
        self.decoder_bridge = nn.Sequential(*([nn.Linear(self.wd_emb_dim, dbridge_dim)] +
                                              [nn.Linear(dbridge_dim, dbridge_dim) for i in range(dbridge_layer - 1)]))
        self.memory_keys = [
            nn.Parameter(
                torch.randn((self.wd_emb_dim, len(word_embeddings))),
                requires_grad=True)
            for i in range(memory_hops)]
        # [num_vocab, wd_emb_dim]
        # self.memory_values = [
        # 	nn.Parameter(
        # 		torch.from_numpy(word_embeddings),
        # 		requires_grad=False)
        # 	for i in range(memory_hops)]
        for i in range(memory_hops):
            self.register_parameter("memory_keys_%d" % i, self.memory_keys[i])
            self.memory_keys[i].requires_grad = True
            # self.register_parameter("memory_values_%d" % i, self.memory_values[i])
            # self.memory_values[i].requires_grad = False


    def forward(self, logits, kwd_mask=None):
        if kwd_mask is None:
            feature = self.mlp(logits)
        else:
            # use sampling or provided
            feature = self.mlp(logits * kwd_mask)

        # use attentional memory retrieval here
        for i in range(self.memory_hops):
            attn_weight = torch.softmax(torch.matmul(feature, self.memory_keys[i]), 1)
            # retrived = torch.matmul(attn_weight, self.memory_values[i])
            retrived = torch.matmul(attn_weight, self.embedding.weight.data)
            feature = feature + retrived

        e_feature = self.encoder_bridge(feature)
        d_feature = self.decoder_bridge(feature)
        return e_feature, d_feature
