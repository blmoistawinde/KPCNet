import argparse
import pickle as p
import sys
import json
import torch
import scipy.sparse
from read_data import *
from model.encoderRNN import *
from model.attnDecoderRNN import *
from model.kwdPredictor import *
from model.kwdBridge import *
from beam import *
from evaluate import eval_kwd_out, get_cluster_kwds
from constants import *
from hparams import hparams
from utils import *
import numpy as np
import _locale
_locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])

def main(args):
    print('Enter main')
    word_embeddings = p.load(open(args.word_embeddings, 'rb'))
    print(('Loaded emb of size %d' % len(word_embeddings)))
    word_embeddings = np.array(word_embeddings)
    word2index = p.load(open(args.vocab, 'rb'))
    index2word = reverse_dict(word2index)
    index2kwd, kwd2index, index2cnt = read_kwd_vocab(args.kwd_vocab)
    test_data = read_data(args.test_context, args.test_question, args.test_ids,
                          args.max_post_len, args.max_ques_len, mode='test')

    print('No. of test_data %d' % len(test_data))
    if args.eval_kwd:
        run_eval_kwd(test_data, word_embeddings, word2index, index2word, kwd2index, index2kwd, args)
    else:
        run_model(test_data, word_embeddings, word2index, index2word, kwd2index, index2kwd, args)


def run_model(test_data, word_embeddings, word2index, index2word, kwd2index, index2kwd, args):
    print('Preprocessing test data..')
    hparams.USER_FILTER = (args.load_filter_dir != "")
    not hparams.USER_FILTER
    q_test_data = preprocess_data(test_data, word2index, kwd2index, hparams.MAX_POST_LEN,
                                  hparams.MAX_QUES_LEN, None,
                                  extract_kwd=not hparams.USER_FILTER,
                                  filter_dir=args.load_filter_dir)
    q_test_data = [np.array(x) for x in q_test_data]
    print('Defining encoder decoder models')
    encoder = EncoderRNN(hparams.HIDDEN_SIZE, word_embeddings, hparams.RNN_LAYERS,
                         dropout=hparams.DROPOUT, update_wd_emb=hparams.UPDATE_WD_EMB)
    decoder = AttnDecoderRNN(hparams.HIDDEN_SIZE, len(word2index), word_embeddings, hparams.ATTN_TYPE,
                             hparams.RNN_LAYERS, dropout=hparams.DROPOUT, update_wd_emb=hparams.UPDATE_WD_EMB,
                             condition=hparams.DECODER_CONDITION_TYPE)
    kwd_predictor = get_predictor(word_embeddings, hparams)
    if not hparams.WITH_MEMORY:
        kwd_bridge = MLPBridge(hparams.HIDDEN_SIZE, hparams.MAX_KWD, hparams.HIDDEN_SIZE, len(word_embeddings[0]),
                               norm_type=hparams.BRIDGE_NORM_TYPE, dropout=hparams.DROPOUT)
    else:
        kwd_bridge = MemoryBridge(word_embeddings, hparams.MAX_KWD, hparams.HIDDEN_SIZE, len(word_embeddings[0]),
                                  memory_hops=hparams.MEMORY_HOPS)
    if hparams.USE_CUDA:
        encoder.cuda()
        decoder.cuda()
        kwd_predictor.cuda()
        kwd_bridge.cuda()

    # Load encoder, decoder params
    print('Loading encoded, decoder params')
    if hparams.USE_CUDA:
        models = torch.load(args.load_models_dir)
    else:
        models = torch.load(args.load_models_dir, map_location='cpu')

    encoder.load_state_dict(models["encoder"])
    decoder.load_state_dict(models["decoder"])
    kwd_predictor.load_state_dict(models["kwd_predictor"])
    kwd_bridge.load_state_dict(models["kwd_bridge"])

    model_prefix = args.load_models_dir[args.load_models_dir.rfind("/")+1:args.load_models_dir.rfind(".")]
    out_prefix = hparams.get_decode_name(model_prefix)
    with torch.no_grad():
        if args.diverse_beam:
            evaluate_diverse_beam(word2index, index2word, encoder, decoder, kwd_predictor, kwd_bridge, q_test_data,
                          args.max_ques_len, args.out_dir, out_prefix, index2kwd, args.save_all_beam)
        elif hparams.SAMPLE_DECODE_WORD:
            evaluate_sample(word2index, index2word, encoder, decoder, kwd_predictor, kwd_bridge, q_test_data,
                            args.max_ques_len, args.out_dir, out_prefix, index2kwd, args.sample_times)
        elif hparams.CLUSTER_KWD:
            kwd_edge_cnt = scipy.sparse.load_npz(args.load_kwd_edge_dir)
            print("Doing kwd clustering")
            kwd_clusters = get_cluster_kwds(kwd_predictor, q_test_data, kwd_edge_cnt, index2kwd, kwd2index)
            hparams.DECODE_USE_KWD_LABEL = True   # kwd label provided by clustering result
            out_prefix = hparams.get_decode_name(model_prefix)
            # select sample_times group out of BEAM_SIZE
            for i in range(args.sample_times):
                q_test_data[5] = kwd_clusters[i]
                evaluate_beam(word2index, index2word, encoder, decoder, kwd_predictor, kwd_bridge, q_test_data,
                              args.max_ques_len, args.out_dir, out_prefix + ".a%d" % i, index2kwd, args.save_all_beam)
        else:
            if args.sample_times < 0:
                evaluate_beam(word2index, index2word, encoder, decoder, kwd_predictor, kwd_bridge, q_test_data,
                              args.max_ques_len, args.out_dir, out_prefix, index2kwd, args.save_all_beam)
            else:
                for i in range(args.sample_times):
                    evaluate_beam(word2index, index2word, encoder, decoder, kwd_predictor, kwd_bridge, q_test_data,
                                  args.max_ques_len, args.out_dir, out_prefix+".a%d" % i, index2kwd, args.save_all_beam)


def run_eval_kwd(test_data, word_embeddings, word2index, index2word, kwd2index, index2kwd, args):
    print('Preprocessing test data..')
    q_test_data = preprocess_data(test_data, word2index, kwd2index, hparams.MAX_POST_LEN,
                                  hparams.MAX_QUES_LEN, None)
    q_test_data = [np.array(x) for x in q_test_data]
    print('Defining model')
    kwd_predictor = get_predictor(word_embeddings, hparams)

    if hparams.USE_CUDA:
        kwd_predictor.cuda()

    # Load encoder, decoder params
    print('Loading encoded, decoder params')
    if hparams.USE_CUDA:
        kwd_predictor.load_state_dict(torch.load(args.kwd_model_dir))
    else:
        kwd_predictor.load_state_dict(torch.load(args.kwd_model_dir, map_location='cpu'))

    kwd_model_name = args.kwd_model_dir[args.kwd_model_dir.rfind("/")+1:]
    eval_kwd_out(kwd_predictor, q_test_data, index2kwd, args.out_dir, kwd_model_name)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--test_context", type=str)
    argparser.add_argument("--test_question", type=str)
    argparser.add_argument("--test_ids", type=str)
    argparser.add_argument("--vocab", type=str)
    argparser.add_argument("--word_embeddings", type=str)
    argparser.add_argument("--kwd_vocab", type=str)
    argparser.add_argument("--kwd_model_dir", type=str)
    argparser.add_argument("--load_models_dir", type=str, default="./ckpt")
    argparser.add_argument("--load_hparams_dir", type=str, default="")
    argparser.add_argument("--load_kwd_edge_dir", type=str, default="./data/kwd_edges.npz")
    argparser.add_argument("--load_filter_dir", type=str, default="")
    argparser.add_argument("--out_dir", type=str, default="./output")
    argparser.add_argument("--eval_kwd", action="store_true")
    argparser.add_argument("--save_all_beam", action="store_true")
    argparser.add_argument("--sample_times", type=int, default=-1)
    hparams.register_arguments(argparser)
    args = argparser.parse_args()
    hparams.update(args)
    if len(args.load_hparams_dir) != 0:
        hparams.load(args.load_hparams_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    print(args)
    main(args)
