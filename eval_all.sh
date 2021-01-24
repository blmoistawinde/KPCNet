#!/bin/bash

SITENAME=Home_and_Kitchen
CQ_DATA_DIR=./data
BLEU_SCRIPT="./eval/multi-bleu.perl"

TEXT_DIR0=$1
RM_TOK_DIR="./eval/remove_tokens.py"
python $RM_TOK_DIR --text $TEXT_DIR0
TEXT_DIR=$TEXT_DIR0"_2"

KWD_EVAL_DIR="./eval/kwd_eval.py"
KWD_DIR="./data/test_ref.kwds"

# BLEU

$BLEU_SCRIPT $CQ_DATA_DIR/test_ref < $TEXT_DIR

# Diversity

export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

NGRAMS_SCRIPT=./eval/all_ngrams.pl

count_uniq_trigrams=$( cat $TEXT_DIR | $NGRAMS_SCRIPT 3 | sort | uniq -c | sort -gr | wc -l )
count_all_trigrams=$( cat $TEXT_DIR | $NGRAMS_SCRIPT 3 | sort | sort -gr | wc -l )
echo -n "Distinct-3 "
echo "scale=4; $count_uniq_trigrams / $count_all_trigrams" | bc

count_uniq_bigrams=$( cat $TEXT_DIR | $NGRAMS_SCRIPT 2 | sort | uniq -c | sort -gr | wc -l )
count_all_bigrams=$( cat $TEXT_DIR | $NGRAMS_SCRIPT 2 | sort | sort -gr | wc -l )
echo -n ", Distinct-2 "
echo "scale=4; $count_uniq_bigrams / $count_all_bigrams" | bc

count_uniq_unigrams=$( cat $TEXT_DIR | $NGRAMS_SCRIPT 1 | sort | uniq -c | sort -gr | wc -l )
count_all_unigrams=$( cat $TEXT_DIR | $NGRAMS_SCRIPT 1 | sort | sort -gr | wc -l )
echo -n ", Distinct-1 "
echo "scale=4; $count_uniq_unigrams / $count_all_unigrams" | bc

# KWD
python $KWD_EVAL_DIR --truth $KWD_DIR \
                   --pred $TEXT_DIR 