import torch
import numpy as np
import random
import math
import time
import re
import html

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def iterate_minibatches(id_seqs, input_seqs, *args, batch_size=128, shuffle=True):
    if shuffle:
        indices = np.arange(len(input_seqs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(input_seqs) - batch_size + 1, batch_size):
        if shuffle:
            ex = indices[start_idx:start_idx + batch_size]
        else:
            ex = slice(start_idx, start_idx + batch_size)
        yield (id_seqs[ex], input_seqs[ex]) + tuple(x[ex] for x in args)


def reverse_dict(word2index):
    index2word = {}
    for w, ix in word2index.items():
        index2word[ix] = w
    return index2word

def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def clean_html(text):
    text = re.sub(r"& (\S+) ;", r"&\1;", text)
    text = re.sub(r"& # (\S+) ;", r"&#\1;", text)
    text = html.unescape(text)
    return text

def clean_text(text):
    return re.sub(r"( <EOS>)", "", clean_html(text.strip()))

# Jaccard
def sent_sim(text1, text2):
    words1, words2 = set(text1.lower().strip().split()), set(text2.lower().strip().split())
    return len(words1 & words2) / len(words1 | words2)

def deduplicate(texts0, preserve=3, threshold=0.5):
    texts = texts0[:]
    assert len(texts) >= preserve and preserve > 0
    if len(texts) == preserve:
        return list(range(len(texts))), texts
    sel_ids, remain_ids = [0], list(range(1, len(texts)))
    sel_texts, remain_texts = texts[:1], texts[1:]
    for i in range(1, preserve):
        overlaps = []
        sel_cand = None
        for cand_id, cand in enumerate(remain_texts):
            overlap = max(sent_sim(cand, sel) for sel in sel_texts)
            if overlap < threshold:
                sel_cand = cand_id
                break
            overlaps.append(overlap)
        if sel_cand is None:
            sel_cand = np.argmin(overlaps)
        sel_texts.append(remain_texts[sel_cand])
        sel_ids.append(remain_ids[sel_cand])
        del remain_texts[sel_cand]
        del remain_ids[sel_cand]
    return sel_ids, sel_texts

def make_filter_mask(post, filter_dict, kwd2index):
    curr_kwd_filter_mask = [0 for i in range(len(kwd2index))]
    for keys, to_filters in filter_dict.items():
        if keys.startswith("@") and keys.endswith("@"):  # regex
            if bool(re.search(keys[1:-1], post)):
                for kwd0 in to_filters:
                    curr_kwd_filter_mask[kwd2index[kwd0]] = -1e20
        else:
            for k in keys.split(","):
                if k in post:
                    for kwd0 in to_filters:
                        curr_kwd_filter_mask[kwd2index[kwd0]] = -1e20
    return curr_kwd_filter_mask