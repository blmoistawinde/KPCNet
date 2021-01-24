from constants import *
import unicodedata
import numpy as np
from collections import defaultdict
from hparams import hparams
from utils import *


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s, max_len):
    #s = unicode_to_ascii(s.lower().strip())
    s = s.lower().strip()
    # words = s.split()
    # s = ' '.join(words[:max_len])
    return s


def get_context(line, max_post_len, max_ques_len):
    context = normalize_string(line, max_post_len-1)
    return context


def read_data(context_fname, question_fname, ids_fname,
              max_post_len, max_ques_len, count=None, mode='train'):
    if ids_fname is not None:
        ids = []
        for line in open(ids_fname, 'r').readlines():
            curr_id = line.strip('\n')
            ids.append(curr_id)

    print("Reading lines...")
    data = []
    i = 0
    for line in open(context_fname, 'r').readlines():
        context = get_context(line, max_post_len, max_ques_len)
        if ids_fname is not None:
            data.append([ids[i], context, None])
        else:
            data.append([None, context, None])
        i += 1
        if count and i == count:
            break
    i = 0
    for line in open(question_fname, 'r').readlines():
        question = normalize_string(line, max_ques_len-1)
        data[i][2] = question
        i += 1
        if count and i == count:
            break
    assert(i == len(data))

    if ids_fname is not None:
        updated_data = []
        i = 0
        if mode == 'test':
            max_per_id_count = 1
        else:
            max_per_id_count = len(data)
        data_ct_per_id = defaultdict(int)
        for curr_id in ids:
            data_ct_per_id[curr_id] += 1
            if data_ct_per_id[curr_id] <= max_per_id_count:
                updated_data.append(data[i])
            i += 1
            if count and i == count:
                break
        assert (i == len(data))
        return updated_data

    return data

def read_kwd_vocab(fname):
    with open(fname, encoding="utf-8") as f:
        raw_kwds = []
        for line in f:
            tmp = line.strip().split()
            kwd, cnt = tmp[0], int(tmp[1])
            raw_kwds.append((kwd, cnt))
    index2kwd = [kwd for kwd, cnt in raw_kwds[:hparams.MAX_KWD]]
    index2cnt = [cnt for kwd, cnt in raw_kwds[:hparams.MAX_KWD]]
    kwd2index = {kwd: i for i, kwd in enumerate(index2kwd)}
    return index2kwd, kwd2index, index2cnt

def read_kwds(fname, kwd2index, count=None):
    all_kwd_ids = []
    with open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if count is not None and i >= count:
                break
            kwds = line.strip().split()
            kwd_ids = [kwd2index[kwd] for kwd in kwds if kwd in kwd2index]
            all_kwd_ids.append(kwd_ids)
    return all_kwd_ids