import argparse
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--truth", type=str, default="../data/test_ref.kwds")
    parser.add_argument("--kwd", type=str)
    args = parser.parse_args()

    f_truth = open(args.truth, encoding="utf-8")
    kwd_fname = args.kwd
    f_kwd = open(kwd_fname, encoding="utf-8")

    p_at_5 = []
    p_at_10 = []
    p_at_20 = []
    for truth, kwd_line in tqdm(zip(f_truth, f_kwd), total=2304):
        tmp = kwd_line.strip().split()
        kwds = [kwd for i, kwd in enumerate(tmp) if i % 2 == 0]
        kwds_truth = set(truth.strip().split())
        p_at_5.append(len(set(kwds[:5]) & kwds_truth))
        p_at_10.append(len(set(kwds[:10]) & kwds_truth))
        p_at_20.append(len(set(kwds[:20]) & kwds_truth))
    p_at_5 = np.array(p_at_5)
    p_at_10 = np.array(p_at_10)
    p_at_20 = np.array(p_at_20)
    print(f"P@5: {np.sum(p_at_5)/(5*len(p_at_5)):.3f}")
    print(f"P@10: {np.sum(p_at_10)/(10*len(p_at_10)):.3f}")
    print(f"P@20: {np.sum(p_at_20)/(20*len(p_at_20)):.3f}")
