import argparse
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=str)
    parser.add_argument("--truth_one", type=str)
    parser.add_argument("--kwd", type=str)
    parser.add_argument("--min", type=float, default=0.05)
    parser.add_argument("--max", type=float, default=0.20)
    parser.add_argument("--step", type=float, default=0.005)
    args = parser.parse_args()

    thre = args.min
    best_thre, best_mae = args.min, float("inf")
    while thre <= args.max:
        TPs, FPs, FNs = [], [], []
        num_preds, num_truths = [], []
        f_truth = open(args.truth, encoding="utf-8")
        f_truth_one = open(args.truth_one, encoding="utf-8")
        kwd_fname = args.kwd
        f_kwd = open(kwd_fname, encoding="utf-8")
        for truth, kwd_line, truth_one in zip(f_truth, f_kwd, f_truth_one):
            tmp = kwd_line.strip().split()
            kwds = np.array([kwd for i, kwd in enumerate(tmp) if i % 2 == 0])
            kwds_prob = np.array([float(v[:-1])/100 for i, v in enumerate(tmp) if i % 2])
            kwds_sel = set(kwds[kwds_prob > thre])
            kwds_truth = set(truth.strip().split())
            kwds_truth_one = set(truth_one.strip().split())
            TPs.append(len(kwds_sel & kwds_truth))
            FPs.append(len(kwds_sel - kwds_truth))
            FNs.append(len(kwds_truth - kwds_sel))
            num_preds.append(len(kwds_sel))
            num_truths.append(len(kwds_truth_one))
        micro_f1 = 2*np.sum(TPs) / (np.sum(FPs)+np.sum(FNs)+2*np.sum(TPs))
        micro_rec = np.sum(TPs) / (np.sum(FPs)+np.sum(TPs))
        micro_pre = np.sum(TPs) / (np.sum(FNs)+np.sum(TPs))
        num_preds, num_truths = np.array(num_preds), np.array(num_truths)
        mae = np.sum(np.abs(num_preds-num_truths))/np.sum(num_truths)
        if mae < best_mae:
            best_thre, best_mae = thre, mae
        print(f"thre: {thre:.3f}\tmicro F1: {micro_f1:.3f}\tmicro precision: {micro_pre:.3f}\tmicro recall: {micro_rec:.3f}\tavg_preds:{np.mean(num_preds):.2f}\tMAE:{mae:.3f}")
        thre += args.step
    print(f"best threshole: {best_thre:.3f}, MAE: {best_mae:.3f}")