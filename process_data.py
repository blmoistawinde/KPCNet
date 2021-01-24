import torch
from torch.autograd import Variable
import pdb
import numpy as np
from tqdm import trange, tqdm
from constants import *
import spacy
import json
import re
from prepare.build_kwd_vocab import extract_kwds
from hparams import hparams
from utils import *

# Return a list of indexes, one for each word in the sentence, plus EOS
def prepare_sequence(seq, word2index, max_len):
    sequence = [word2index[w] if w in word2index else word2index['<unk>'] for w in seq.split(' ')[:max_len-1]]
    sequence.append(word2index[EOS_token])
    length = len(sequence)
    sequence += [word2index[PAD_token]]*(max_len - len(sequence))
    return sequence, length

def preprocess_data(records, word2index, kwd2index, max_post_len, max_ques_len, kwd_data_dir="",
                    extract_kwd=True, filter_dir=""):
    all_kwds = np.array(list(kwd2index.keys()))
    id_seqs = []
    post_seqs = []
    post_lens = []
    ques_seqs = []
    ques_lens = []

    extract_kwd = extract_kwd and (kwd_data_dir != "")  # should not be used with filter_mask
    if extract_kwd:
        kwd_labels = []
        # since kwds per example is sparse compared to the whole vocab, we use negative sampling to balance pos/neg
        kwd_masks = []
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    filter_mask = (filter_dir != "")
    if filter_mask:
        with open(filter_dir, encoding="utf-8") as f:
            filter_dict = json.load(f)
        kwd_filter_masks = []  # the mask here is for filter out kwds

    for i in trange(len(records)):
        curr_id, post, ques = records[i]
        id_seqs.append(curr_id)
        if filter_mask:
            curr_kwd_filter_mask = make_filter_mask(post, filter_dict, kwd2index)
            kwd_filter_masks.append(curr_kwd_filter_mask)
        if extract_kwd:
            kwds = extract_kwds(nlp, ques)
            curr_kwd_label = [0 for i in range(len(kwd2index))]
            curr_kwd_mask = [0 for i in range(len(kwd2index))]
            for kwd in kwds:
                if kwd in kwd2index:
                    curr_kwd_label[kwd2index[kwd]] = 1
                    curr_kwd_mask[kwd2index[kwd]] = 1
            # negative sampling
            num_negs = min(len(kwd2index) - len(kwds), max(hparams.MIN_NEG_KWD, len(kwds) * hparams.NEG_KWD_PER))
            neg_kwds = set()
            while len(neg_kwds) < num_negs:
                sample = np.random.choice(all_kwds)
                if sample not in (neg_kwds | kwds):
                    neg_kwds.add(sample)
                    curr_kwd_mask[kwd2index[sample]] = 1
            kwd_labels.append(curr_kwd_label)
            kwd_masks.append(curr_kwd_mask)
        # truncate here to preserve complete context for previous steps
        post = " ".join(post.split()[:max_post_len - 1])
        ques = " ".join(ques.split()[:max_ques_len - 1])
        post_seq, post_len = prepare_sequence(post, word2index, max_post_len)
        post_seqs.append(post_seq)
        post_lens.append(post_len)
        if ques is not None:
            ques_seq, ques_len = prepare_sequence(ques, word2index, max_ques_len)
            ques_seqs.append(ques_seq)
            ques_lens.append(ques_len)
    if extract_kwd:
        return id_seqs, post_seqs, post_lens, ques_seqs, ques_lens, kwd_labels, kwd_masks
    elif filter_mask:
        return id_seqs, post_seqs, post_lens, ques_seqs, ques_lens, kwd_filter_masks, kwd_filter_masks  # -2 just as a placeholder
    else:
        return id_seqs, post_seqs, post_lens, ques_seqs, ques_lens

# extract kwd in running
def build_kwd_arr(kwds, kwd2index):
    N_samples, N_kwds = len(kwds), len(kwd2index)
    all_kwds = np.array(list(kwd2index.values()))
    kwds_arr = np.zeros((N_samples, N_kwds))
    kwds_mask_arr = np.zeros((N_samples, N_kwds))
    for i, curr_kwds in enumerate(kwds):
        curr_kwds = set(curr_kwds)
        for kwd0 in curr_kwds:
            kwds_arr[i, kwd0] = 1
            kwds_mask_arr[i, kwd0] = 1
        num_negs = min(N_kwds - len(curr_kwds), max(hparams.MIN_NEG_KWD, len(curr_kwds) * hparams.NEG_KWD_PER))
        neg_kwds = np.random.choice(all_kwds, size=num_negs+len(curr_kwds), replace=False)
        neg_kwds = list(set(neg_kwds)-curr_kwds)[:num_negs]
        kwds_mask_arr[i, neg_kwds] = 1
    return kwds_arr, kwds_mask_arr


