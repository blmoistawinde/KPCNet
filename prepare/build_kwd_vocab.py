import spacy
from tqdm import tqdm
from collections import defaultdict

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
    nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
    in_file = "../data/train_ques.txt"
    out_file = "../data/train_kwd_vocab.txt"
    THRESHOLD = 3

    kwd_cnt = defaultdict(int)
    with open(in_file, encoding="utf-8") as f:
        for line in tqdm(f, total=143860):
            kwds = extract_kwds(nlp, line.strip())
            for wd in kwds:
                kwd_cnt[wd] += 1

    print(f"Original kwds: {len(kwd_cnt)}")
    kwds_filtered = sorted([(k,v) for (k,v) in kwd_cnt.items() if v >= THRESHOLD], key=lambda x: x[1], reverse=True)
    print(f"Filtered kwds: {len(kwds_filtered)}")
    
    with open(out_file, "w", encoding="utf-8") as f:
        for k, v in kwds_filtered:
            f.write(f"{k}\t{v}\n")
