import argparse
import os
import pickle as p
import string
import time
import datetime
import math
import json
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy

import numpy as np

from read_data import *
from process_data import *
from run import *
from constants import *
from hparams import hparams
from utils import *
import _locale
_locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])


def main(args):
    word_embeddings = p.load(open(args.word_embeddings, 'rb'))
    word_embeddings = np.array(word_embeddings)
    word2index = p.load(open(args.vocab, 'rb'))

    index2kwd, kwd2index, index2cnt = read_kwd_vocab(args.kwd_vocab)
    if hparams.BALANCE_KWD_CLASS:
        # adjust weight for different kwd class based on median freqency
        index2cnt = np.array(index2cnt)
        base_freq = np.median(index2cnt)
        kwd_weight = np.sqrt(base_freq/index2cnt)
        kwd_weight = torch.FloatTensor(kwd_weight)
        if hparams.USE_CUDA:
            kwd_weight = kwd_weight.cuda()
    else:
        kwd_weight = None

    subset_count = args.subset_count if args.subset_count > 0 else None
    train_data = read_data(args.train_context, args.train_question, args.train_ids,
                            args.max_post_len, args.max_ques_len, subset_count)
    test_data = read_data(args.tune_context, args.tune_question, args.tune_ids,
                          args.max_post_len, args.max_ques_len, subset_count)

    if args.kwd_data_dir:  # load pre-extracted kwd, save time in training
        print(f"load kwds from {args.kwd_data_dir}")
        train_kwds = read_kwds(os.path.join(args.kwd_data_dir, "train.kwds"), kwd2index, subset_count)
        test_kwds = read_kwds(os.path.join(args.kwd_data_dir, "tune.kwds"), kwd2index, subset_count)
        assert len(train_kwds) == len(train_data), print(len(train_kwds), len(train_data))
        assert len(test_kwds) == len(test_data)
    else:
        train_kwds, test_kwds = None, None

    print('No. of train_data %d' % len(train_data))
    print('No. of test_data %d' % len(test_data))

    print("Preprocessing train")
    q_train_data = preprocess_data(train_data, word2index, kwd2index,
                                   args.max_post_len, args.max_ques_len, args.kwd_data_dir, extract_kwd=False)
    q_train_data = [np.array(x) for x in q_train_data]
    print("Preprocessing val")
    q_test_data = preprocess_data(test_data, word2index, kwd2index,
                                  args.max_post_len, args.max_ques_len, args.kwd_data_dir, extract_kwd=False)
    q_test_data = [np.array(x) for x in q_test_data]

    if args.pretrain_ques:
        run_seq2seq(q_train_data, q_test_data, word2index, word_embeddings,
                    hparams.MAX_QUES_LEN, kwd_weight, not args.freeze_kwd_model,
                    train_kwds, test_kwds, kwd2index, args.kwd_model_dir, args.save_dir, args.load_models_dir)
    elif args.pretrain_kwd:
        run_kwd(q_train_data, q_test_data, index2kwd, word_embeddings, kwd_weight,
            train_kwds, test_kwds, kwd2index, args.save_dir)
    else:
        print('Please specify model to pretrain')
        return


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train_context", type = str)
    argparser.add_argument("--train_question", type = str)
    argparser.add_argument("--train_ids", type=str)
    argparser.add_argument("--tune_context", type = str)
    argparser.add_argument("--tune_question", type = str)
    argparser.add_argument("--tune_answer", type = str)
    argparser.add_argument("--tune_ids", type=str)
    argparser.add_argument("--test_context", type = str)
    argparser.add_argument("--test_question", type = str)
    argparser.add_argument("--test_answer", type = str)
    argparser.add_argument("--test_ids", type=str)
    argparser.add_argument("--save_dir", type=str, default="")
    argparser.add_argument("--vocab", type = str)
    argparser.add_argument("--word_embeddings", type = str)
    argparser.add_argument("--pretrain_ques", action="store_true")
    argparser.add_argument("--kwd_vocab", type = str)
    argparser.add_argument("--pretrain_kwd", action="store_true")
    argparser.add_argument("--kwd_data_dir", type = str)
    argparser.add_argument("--kwd_model_dir", type = str)
    argparser.add_argument("--load_models_dir", type=str, default=None)
    argparser.add_argument("--save_hparams_dir", type=str, default="")
    argparser.add_argument("--subset_count", type=int, default=-1)
    hparams.register_arguments(argparser)
    args = argparser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    hparams.update(args)
    if args.pretrain_kwd:
        hparams.MODEL_TYPE = "kwd"
    if len(args.save_hparams_dir) > 0:
        os.makedirs(args.save_hparams_dir, exist_ok=True)
        if args.pretrain_kwd:
            save_hparams_dir = os.path.join(args.save_hparams_dir, hparams.get_exp_name() + ".kwd_pred.json")
        else:
            save_hparams_dir = os.path.join(args.save_hparams_dir, hparams.get_exp_name() + ".json")
        hparams.save(save_hparams_dir)
    print(args)
    print("")
    main(args)

