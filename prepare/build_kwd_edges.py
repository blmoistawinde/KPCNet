import numpy as np
import scipy.sparse
from read_data import read_kwd_vocab
from itertools import combinations
from collections import defaultdict

if __name__ == "__main__":
    index2kwd, kwd2index, index2cnt = read_kwd_vocab("../data/train_kwd_vocab.txt")
    N = len(index2kwd)
    cnt_dict = defaultdict(int)
    rows, cols, datas = [], [], []
    for line in open("../data/train.kwds", encoding="utf-8"):
        words = line.strip().split()
        for a, b in combinations(words, 2):
            if a in kwd2index and b in kwd2index:
                cnt_dict[(kwd2index[a], kwd2index[b])] += 1
    for (row, col), cnt in cnt_dict.items():
        rows.append(row)
        cols.append(col)
        datas.append(cnt)
    kwd_edges = scipy.sparse.csr_matrix((datas, (rows, cols)), shape=(N, N))
    kwd_edges = (kwd_edges + kwd_edges.T)
    scipy.sparse.save_npz("../data/kwd_edges.npz", kwd_edges)
    kwd_edges = scipy.sparse.load_npz("../data/kwd_edges.npz")