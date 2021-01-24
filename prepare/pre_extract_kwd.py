import spacy
import argparse
import numpy as np
from tqdm import tqdm

my_stopwords = {"use", "work", "thank", "need", "know", "want", "product", "buy", "say",
"good", "-", 's', 'amp'}
# 3 POS version
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
    parser.add_argument("--split", type=str, default='test')
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
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
        fo = open(f"{args.truth_dir}/{args.split}_ref.kwds", "w", encoding="utf-8")
        all_ref_names = [f"{args.truth_dir}/{args.split}_ref{i}" for i in range(10)]
        all_ref_files = [open(fname, encoding="utf-8") for fname in all_ref_names]
        for refs in tqdm(zip(*all_ref_files), total=2304):
            curr_kwds = set()
            for ref0 in refs:
                curr_kwds |= extract_kwds(nlp, ref0.strip())
            fo.write("\t".join(curr_kwds) + "\n")
        fo.close()
        pass
        