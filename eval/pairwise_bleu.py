"""
Adapted from https://github.com/pytorch/fairseq/blob/master/examples/translation_moe/score.py
Scoring script for computing pairwise BLEU and multi-ref BLEU over a set of
candidate hypotheses.

See `"Mixture Models for Diverse Machine Translation: Tricks of the Trade"
(Shen et al., 2019) <https://arxiv.org/abs/1902.07816>`_.
"""

import argparse
import numpy as np
from sacrebleu import compute_bleu, corpus_bleu as _corpus_bleu
import re
import os
import random
from glob import glob
from itertools import chain

# Jaccard
def sent_sim(text1, text2):
    words1, words2 = set(text1.lower().strip().split()), set(text2.lower().strip().split())
    return len(words1 & words2) / len(words1 | words2)

def deduplicate(texts, preserve=3, threshold=0.5):
    assert len(texts) >= preserve and preserve > 0
    if len(texts) == preserve:
        return list(range(len(texts))), texts
    sel_ids, remain_ids = [0], list(range(len(texts)))
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

def corpus_bleu(sys_stream, ref_streams):
    bleu = _corpus_bleu(sys_stream, ref_streams, tokenize='none')
    return bleu.score

def sentence_bleu(hypothesis, reference):
    bleu = _corpus_bleu(hypothesis, reference)
    for i in range(1, 4):
        bleu.counts[i] += 1
        bleu.totals[i] += 1
    bleu = compute_bleu(
        bleu.counts, bleu.totals,
        bleu.sys_len, bleu.ref_len, smooth_method='exp'
    )
    return bleu.score

def pairwise(sents):
    _ref, _hypo = [], []
    for s in sents:
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    _ref.append(s[i])
                    _hypo.append(s[j])
    return corpus_bleu(_hypo, [_ref])

def clean_text(text):
    return re.sub(r"( <unk>| <EOS>)", "", text)

def intra_ref(refs):
    print('ref pairwise BLEU: %.2f' % pairwise(refs))
    refs = list(zip(*refs))
    m = len(refs)
    concat_h = []
    # [num_ref-1, num_sents*num_ref]
    concat_rest = [[] for j in range(m - 1)]
    for i, h in enumerate(refs):
        rest = refs[:i] + refs[i+1:]
        concat_h.append(h)
        for j in range(m - 1):
            concat_rest[j].extend(rest[j])

    concat_h = list(chain.from_iterable(concat_h))
    bleu = corpus_bleu(concat_h, concat_rest)
    print('multi-reference BLEU (leave-one-out): %.2f' % bleu)

def multi_ref(refs, hypos):
    _ref, _hypo = [], []
    ref_cnt = 0
    assert len(refs) == len(hypos)

    for rs, hs in zip(refs, hypos):
        a = set()
        for h in hs:
            s = [sentence_bleu(h, r) for r in rs]
            j = np.argmax(s)
            _ref.append(rs[j])
            _hypo.append(h)
            best = [k for k in range(len(rs)) if s[k] == s[j]]
            a.add(random.choice(best))
        ref_cnt += len(a)
    print('#refs covered: %.2f' % (ref_cnt / len(refs)))

    # transpose refs and hypos
    refs = list(zip(*refs))
    hypos = list(zip(*hypos))

    # compute multi-ref corpus BLEU (leave-one-out to be comparable to intra_ref)
    k = len(hypos)
    m = len(refs)
    # flatten to [num_hypo*num_sents]
    flat_hypos = [hypos[j][i] for i in range(len(hypos[0])) for j in range(k)]
    # [num_ref, num_hypo*num_sents], duplicate num_hypo times
    duplicated_refs = [
        [ref for ref in refs_i for _ in range(k)]
        for refs_i in refs
    ]
    loo_bleus = []
    # average across m leave-one-out corpus
    for held_out_ref in range(m):
        remaining_refs = duplicated_refs[:held_out_ref] + duplicated_refs[held_out_ref+1:]
        assert len(remaining_refs) == m - 1
        loo_bleus.append(corpus_bleu(flat_hypos, remaining_refs))
    print('average multi-reference BLEU (leave-one-out): %.2f' % np.mean(loo_bleus))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_prefix", type=str, default="")
    parser.add_argument("--ref_prefix", type=str, default="")
    parser.add_argument("--deduplicate", action="store_true")
    args = parser.parse_args()

    BEAM_SIZE = 6
    GROUP_NUM = 2
    TAKE_NUM = 3

    if args.deduplicate:
        print("Do deduplication")
    else:
        print("Not do deduplication")
    id2hyps = []
    id2refs = []
    # load hyps
    prefix = args.hyp_prefix
    id0 = prefix.rfind("/")+1
    dir, fprefix = prefix[:id0], prefix[id0:]
    hypid = 0
    for fname in sorted(os.listdir(dir)):
        if not fname.startswith(fprefix):
            continue
        if fprefix.endswith("a"):
            if not bool(re.search(r"a[0-9]+\.beam[0-9]+$", fname)):
                continue
        else:
            if not bool(re.search(r"\.beam[0-9]+$", fname)) or bool(re.search(r"a[0-9]+\.beam[0-9]+$", fname)):
                continue
        # print(hypid, fname)
        with open(dir+fname, encoding="utf-8") as f:
            for lineid, line in enumerate(f):
                sent = line.strip()
                if hypid == 0:
                    id2hyps.append([clean_text(sent)])
                else:
                    id2hyps[lineid].append(clean_text(sent))
        hypid += 1
    
    for lineid, hyps0 in enumerate(id2hyps):
        if fprefix.endswith("a"):
            # adjust order to let beam first
            hyps = [hyps0[(i//GROUP_NUM+(i % GROUP_NUM)*BEAM_SIZE)] for i in range(len(hyps0))]
            hyps0 = hyps
        if args.deduplicate:
            sel_ids, sel_texts = deduplicate(hyps0, preserve=TAKE_NUM)
            id2hyps[lineid] = sel_texts
        else:
            id2hyps[lineid] = hyps0[:TAKE_NUM]
    
    print('Hypothesis pairwise BLEU: %.2f' % pairwise(id2hyps))
    # load refs
    prefix = args.ref_prefix
    id0 = prefix.rfind("/") + 1
    dir, fprefix = prefix[:id0], prefix[id0:]
    refid = 0
    for fname in os.listdir(dir):
        if not (fname.startswith(fprefix) and bool(re.search(r"[0-9]+$", fname))):
            continue
        # print(refid, dir + fname)
        with open(dir + fname, encoding="utf-8") as f:
            for lineid, line in enumerate(f):
                sent = line.strip()
                if refid == 0:
                    id2refs.append([sent])
                else:
                    id2refs[lineid].append(sent)
        refid += 1
    # intra_ref(id2refs)
    multi_ref(id2refs, id2hyps)