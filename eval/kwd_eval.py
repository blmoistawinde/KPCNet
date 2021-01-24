import spacy
import argparse
import numpy as np
from tqdm import tqdm

my_stopwords = {"use", "work", "thank", "need", "know", "want", "product", "buy", "say",
"good", "-", 's', 'amp'}

def extract_kwds(nlp, text):
    kwds = set()
    for word in nlp(text):
        # if not (word.is_stop or word.pos_ in {"PUNCT", "INTJ", "NUM", "PUNCT", "SYM"}):
        if not word.is_stop and word.lemma_ not in my_stopwords and word.pos_ in {"NOUN", "VERB", "ADJ"}:
            # print(f"{word.text} {word.pos_}")
            kwds.add(word.lemma_)
    return kwds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth_dir", type=str, default="")
    parser.add_argument("--extract_three", type=bool, default=False)
    parser.add_argument("--truth", type=str, default="")
    parser.add_argument("--pred", type=str, default="")
    parser.add_argument("--kwd", type=str, default="")
    parser.add_argument("--kwd_sample", type=str, default="")
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
    if args.truth_dir != "":        # pre extract kwds
        if args.extract_three:
            for split in ["train", "tune", "test"]:
                fi = open(f"{args.truth_dir}/{split}_ques.txt", encoding="utf-8")
                fo = open(f"{args.truth_dir}/{split}.kwds", "w", encoding="utf-8")
                for line in fi:
                    kwds0 = extract_kwds(nlp, line.strip())
                    fo.write("\t".join(kwds0)+"\n")
                fi.close()
                fo.close()
        else:
            # combine 10 test refs
            fo = open(f"{args.truth_dir}/test_ref.kwds", "w", encoding="utf-8")
            all_ref_names = [f"{args.truth_dir}/test_ref{i}" for i in range(10)]
            all_ref_files = [open(fname, encoding="utf-8") for fname in all_ref_names]
            for refs in tqdm(zip(*all_ref_files), total=2304):
                curr_kwds = set()
                for ref0 in refs:
                    curr_kwds |= extract_kwds(nlp, ref0.strip())
                fo.write("\t".join(curr_kwds) + "\n")
            fo.close()
            pass
    else:
        f_truth = open(args.truth, encoding="utf-8")
        f_pred = open(args.pred, encoding="utf-8")
        if args.kwd != "":
            kwd_fname = args.kwd
        else:
            kwd_fname = args.pred[:args.pred.rfind(".")]+".kwds"
        f_kwd = open(kwd_fname, encoding="utf-8")
        if args.kwd_sample != "":
            f_kwd_sample = open(args.kwd_sample, encoding="utf-8")
        else:
            f_kwd_sample = open(kwd_fname[:-1]+"_samples", encoding="utf-8")

        p_at_5 = []
        p_at_10 = []
        p_at_20 = []
        response_nums = []
        predict_nums = []
        sent_lens = []
        
        for truth, kwd_line, kwd_sample, pred in tqdm(zip(f_truth, f_kwd, f_kwd_sample, f_pred), total=2304):
            pred = pred.strip()
            sent_lens.append(len(pred.split()))
            tmp = kwd_line.strip().split()
            kwds = [kwd for i, kwd in enumerate(tmp) if i % 2 == 0]
            kwds_pred = extract_kwds(nlp, pred)
            kwds_truth = set(truth.strip().split())
            kwd_sample = set(kwd_sample.strip().split())
            p_at_5.append(len(set(kwds[:5]) & kwds_truth))
            p_at_10.append(len(set(kwds[:10]) & kwds_truth))
            p_at_20.append(len(set(kwds[:20]) & kwds_truth))
            response_nums.append(len(kwd_sample & kwds_pred))
            predict_nums.append(max(len(kwd_sample), 1))
        p_at_5 = np.array(p_at_5)
        p_at_10 = np.array(p_at_10)
        p_at_20 = np.array(p_at_20)
        response_nums = np.array(response_nums)
        predict_nums = np.array(predict_nums)
        print(f"P@5: {np.sum(p_at_5)/(5*len(p_at_5)):.3f}")
        print(f"P@10: {np.sum(p_at_10)/(10*len(p_at_10)):.3f}")
        print(f"P@20: {np.sum(p_at_20)/(20*len(p_at_20)):.3f}")
        print(f"Macro response: {np.mean(response_nums/predict_nums):.3f}")
        print(f"Micro response: {np.sum(response_nums)/np.sum(predict_nums):.3f}")
        print(f"Avg len: {np.mean(sent_lens):.3f}")
