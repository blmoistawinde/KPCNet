import argparse
import os
import pickle as p
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from read_data import *
from process_data import *
from run import *
from constants import *
from hparams import hparams
from utils import *
from beam import evaluate_beam
import _locale
_locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])

from flask import Flask
from flask import render_template
from flask import request, jsonify, redirect, url_for
from nltk import sent_tokenize, word_tokenize

app = Flask(__name__)
word_embeddings = None
word2index = None
index2word = None
index2kwd, kwd2index, index2cnt = None, None, None
encoder, decoder, kwd_predictor, kwd_bridge = None, None, None, None
filter_dict = None

def text2words(text, max_len=200):
    return [x.lower() for sent in sent_tokenize(text)
                      for x in word_tokenize(text)][:max_len]

def infer(context, method="cluster"):
    assert method in {"beam", "cluster"}
    hparams.BATCH_SIZE = 1
    words0 = text2words(context, hparams.MAX_POST_LEN)
    # batch of size 1
    input_seqs = [[word2index[x] if x in word2index else word2index[UNK_token] for x in words0]]
    input_lens = [len(words0)]
    test_data = [["id0"],input_seqs,input_lens,[None],[None],[0],[0]]
    kwd_filter_mask0 = make_filter_mask(" ".join(words0), filter_dict, kwd2index)
    print(sum(kwd_filter_mask0))
    kwd_filter_masks = [kwd_filter_mask0]  # the mask here is for filter out kwds
    test_data[-1] = kwd_filter_masks
    if method == "beam":
        out_seqs = evaluate_beam(word2index, index2word, encoder, decoder, kwd_predictor, kwd_bridge, test_data, hparams.MAX_QUES_LEN, "./infer_out", "infer", index2kwd, save_all_beam=True, infer=True)
    else:
        hparams.KWD_CLUSTERS = 2
        hparams.DECODE_USE_KWD_LABEL = True
        kwd_edge_cnt = scipy.sparse.load_npz("./data/kwd_edges.npz")
        kwd_clusters = get_cluster_kwds(kwd_predictor, test_data, kwd_edge_cnt, index2kwd, kwd2index)
        out_seqs = []
        for i in range(hparams.KWD_CLUSTERS):
            test_data[5] = kwd_clusters[i]
            tmp_seqs = evaluate_beam(word2index, index2word, encoder, decoder, kwd_predictor, kwd_bridge, test_data, hparams.MAX_QUES_LEN, "./infer_out", "infer", index2kwd, save_all_beam=True, infer=True)
            out_seqs.extend(tmp_seqs)
    cleaned_out_seqs = [clean_text(x) for x in out_seqs]
    _, filtered_texts = deduplicate(cleaned_out_seqs, 3)
    return filtered_texts

@app.route('/', methods = ["GET"])
def index():
    return render_template('index.html')

@app.route('/response', methods=["GET", "POST"])
def response():
    text = request.json["text"]
    q1, q2, q3 = infer(text)
    default_response = f'<p><span style="text-decoration:none;">· {q1}</span></p><p><span style="text-decoration:none;">· {q2}</span></p><p><span style="text-decoration:none;">· {q3}</span></p>'
    # default_response = text
    return default_response

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--vocab", default="./data/vocab.p", type = str)
    argparser.add_argument("--word_embeddings", default="./data/word_embeddings.p", type = str)
    argparser.add_argument("--kwd_vocab", default="./data/train_kwd_vocab.txt", type = str)
    argparser.add_argument("--kwd_filter_dir", default="./data/kwd_filter_dict.json", type = str)
    argparser.add_argument("--load_models_dir", default="./ckpt/s2s_D0.3_cnn_noneg_dropout_replace_fr.epoch59.models", type=str)
    argparser.add_argument("--load_hparams_dir", type=str, default="./hparams/s2s_D0.3_cnn_noneg_dropout_replace_fr.json")
    hparams.register_arguments(argparser)
    args = argparser.parse_args()
    hparams.update(args)
    word_embeddings = p.load(open(args.word_embeddings, 'rb'))
    word_embeddings = np.array(word_embeddings)
    word2index = p.load(open(args.vocab, 'rb'))
    index2word = reverse_dict(word2index)
    index2kwd, kwd2index, index2cnt = read_kwd_vocab(args.kwd_vocab)

    encoder = EncoderRNN(hparams.HIDDEN_SIZE, word_embeddings, hparams.RNN_LAYERS,
                         dropout=hparams.DROPOUT, update_wd_emb=hparams.UPDATE_WD_EMB)
    decoder = AttnDecoderRNN(hparams.HIDDEN_SIZE, len(word2index), word_embeddings, hparams.ATTN_TYPE,hparams.RNN_LAYERS, dropout=hparams.DROPOUT, update_wd_emb=hparams.UPDATE_WD_EMB,
                                condition=hparams.DECODER_CONDITION_TYPE)
    kwd_predictor = get_predictor(word_embeddings, hparams)
    kwd_bridge = MLPBridge(hparams.HIDDEN_SIZE, hparams.MAX_KWD, hparams.HIDDEN_SIZE, len(word_embeddings[0]),
                                norm_type=hparams.BRIDGE_NORM_TYPE, dropout=hparams.DROPOUT)
    if hparams.USE_CUDA:
        encoder.cuda()
        decoder.cuda()
        kwd_predictor.cuda()
        kwd_bridge.cuda()
    models = torch.load(args.load_models_dir)
    hparams.load(args.load_hparams_dir)
    encoder.load_state_dict(models["encoder"])
    decoder.load_state_dict(models["decoder"])
    kwd_predictor.load_state_dict(models["kwd_predictor"])
    kwd_bridge.load_state_dict(models["kwd_bridge"])
    with open(args.kwd_filter_dir, encoding="utf-8") as f:
        filter_dict = json.load(f)

    app.run(host='0.0.0.0', port=10100)
