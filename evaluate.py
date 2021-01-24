import os
from constants import *
from process_data import *
from model.model_utils import *
from hparams import hparams
from utils import *
import torch
import scipy.special
import torch.nn as nn
from torch.autograd import Variable
from itertools import combinations
from collections import defaultdict
from sklearn.cluster import SpectralClustering


def evaluate(test_data, encoder, decoder, kwd_predictor, kwd_bridge, SOS_idx, max_output_length, BATCH_SIZE,
             kwd_weight=None, test_kwds=None, kwd2index=None):
    if test_kwds is None:
        ids_seqs, input_seqs, input_lens, output_seqs, output_lens, kwd_labels, kwd_masks = test_data
    else:
        ids_seqs, input_seqs, input_lens, output_seqs, output_lens = test_data
        kwd_labels, kwd_masks = build_kwd_arr(test_kwds, kwd2index)
    total_loss = 0.
    n_batches = len(input_seqs) // BATCH_SIZE

    with torch.no_grad():
        for ids_seqs_batch, input_seqs_batch, input_lens_batch, output_seqs_batch, output_lens_batch, kwd_labels_batch, kwd_masks_batch in \
                    iterate_minibatches(ids_seqs, input_seqs, input_lens, output_seqs, output_lens, kwd_labels, kwd_masks, batch_size=BATCH_SIZE):
            if hparams.USE_CUDA:
                input_seqs_batch = torch.LongTensor(input_seqs_batch).cuda().transpose(0, 1)
                output_seqs_batch = torch.LongTensor(output_seqs_batch).cuda().transpose(0, 1)
                kwd_labels_batch = torch.FloatTensor(kwd_labels_batch).cuda()
                kwd_masks_batch = torch.FloatTensor(kwd_masks_batch).cuda()
            else:
                input_seqs_batch = torch.LongTensor(input_seqs_batch).transpose(0, 1)
                output_seqs_batch = torch.LongTensor(output_seqs_batch).transpose(0, 1)
                kwd_labels_batch = torch.FloatTensor(kwd_labels_batch)
                kwd_masks_batch = torch.FloatTensor(kwd_masks_batch)

            encoder_outputs, encoder_hidden = encoder(input_seqs_batch, input_lens_batch, None)
            decoder_input = torch.LongTensor([SOS_idx] * BATCH_SIZE)
            decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]
            logits = kwd_predictor(input_seqs_batch, input_lens_batch)
            e_features, d_features = kwd_bridge(logits, kwd_mask=kwd_labels_batch)
            # masked loss
            if not hparams.FREEZE_KWD_MODEL:
                if kwd_weight is None:
                    loss_kwd = torch.nn.BCEWithLogitsLoss()(logits*kwd_masks_batch, kwd_labels_batch)
                else:
                    loss_kwd = torch.nn.BCEWithLogitsLoss(pos_weight=kwd_weight)(logits*kwd_masks_batch, kwd_labels_batch)

            if not hparams.NO_ENCODER_BRIDGE:
                ### Replace SOS token embedding with the features obtained from kwd predictor
                encoder_outputs[0, :, :] = e_features
            all_decoder_outputs = torch.zeros(max_output_length, BATCH_SIZE, decoder.output_size)
            if hparams.USE_CUDA:
                decoder_input = decoder_input.cuda()
                all_decoder_outputs = all_decoder_outputs.cuda()

            # Run through decoder one time step at a time
            for t in range(max_output_length):
                if (not hparams.NO_DECODER_BRIDGE) and ((t == 0 and hparams.DECODER_CONDITION_TYPE == 'replace') or
                        hparams.DECODER_CONDITION_TYPE == 'concat'):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs, d_features)
                else:
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                all_decoder_outputs[t] = decoder_output
                # Choose top word from output
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(1)

            loss_fn = torch.nn.NLLLoss()
            loss_seq2seq = masked_cross_entropy(
                all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
                output_seqs_batch.transpose(0, 1).contiguous(), # -> batch x seq
                output_lens_batch, loss_fn, max_output_length
                )
            loss = loss_seq2seq if hparams.FREEZE_KWD_MODEL else loss_seq2seq + hparams.KWD_LOSS_RATIO * loss_kwd
            total_loss += loss.item()
    return total_loss/n_batches

def evaluate_kwd(index2kwd, kwd_predictor, test_data, out_fname=None, kwd_weight=None, test_kwds=None, kwd2index=None):
    if out_fname:
        out_file = open(out_fname, "w", encoding="utf-8")
    else:
        out_file = None
    if test_kwds is None:
        ids_seqs, input_seqs, input_lens, output_seqs, output_lens, kwd_labels, kwd_masks = test_data
    else:
        ids_seqs, input_seqs, input_lens, output_seqs, output_lens = test_data
        kwd_labels, kwd_masks = build_kwd_arr(test_kwds, kwd2index)
    total_loss = 0.
    n_batches = len(input_seqs) // hparams.BATCH_SIZE

    with torch.no_grad():
        for ids_seqs_batch, input_seqs_batch, input_lens_batch, output_seqs_batch, output_lens_batch, kwd_labels_batch, kwd_masks_batch in \
                    iterate_minibatches(ids_seqs, input_seqs, input_lens, output_seqs, output_lens, kwd_labels, kwd_masks, batch_size=hparams.BATCH_SIZE):
            if hparams.NO_NEG_SAMPLE:
                kwd_masks_batch = torch.ones(kwd_labels_batch.shape)
            if hparams.USE_CUDA:
                input_seqs_batch = torch.LongTensor(input_seqs_batch).cuda().transpose(0, 1)
                kwd_labels_batch = torch.FloatTensor(kwd_labels_batch).cuda()
                kwd_masks_batch = torch.FloatTensor(kwd_masks_batch).cuda()
            else:
                input_seqs_batch = torch.LongTensor(input_seqs_batch).transpose(0, 1)
                kwd_labels_batch = torch.FloatTensor(kwd_labels_batch)
                kwd_masks_batch = torch.FloatTensor(kwd_masks_batch)

            logits = kwd_predictor(input_seqs_batch, input_lens_batch)
            # masked loss
            if kwd_weight is None:
                loss_kwd = torch.nn.BCEWithLogitsLoss()(logits*kwd_masks_batch, kwd_labels_batch)
            else:
                loss_kwd = torch.nn.BCEWithLogitsLoss(pos_weight=kwd_weight)(logits*kwd_masks_batch, kwd_labels_batch)

            if out_file:
                probs = torch.sigmoid(logits).cpu().detach().numpy()
                for prob in probs:
                    top_kwd_ids = np.argsort(prob)[::-1][:hparams.SHOW_TOP_KWD]
                    top_prob = prob[top_kwd_ids]
                    out_file.write("\t".join(f"{index2kwd[i]}\t{prob0:.2%}" for (i, prob0) in zip(top_kwd_ids, top_prob))+"\n")
            loss = loss_kwd
            total_loss += loss.item()
        if out_file:
            out_file.close()
    return total_loss/n_batches

def eval_kwd_out(kwd_predictor, test_data, index2kwd, out_dir, kwd_model_name):
    id_seqs, input_seqs, input_lens, output_seqs, output_lens, kwd_labels, kwd_masks = test_data
    kwd_predictor.eval()
    out_kwds = open(os.path.join(out_dir, kwd_model_name+".kwd_prob"), "w")
    n_batches = len(input_seqs)//hparams.BATCH_SIZE
    for id_seqs_batch, input_seqs_batch, input_lens_batch, kwd_labels_batch in \
            tqdm(iterate_minibatches(id_seqs, input_seqs, input_lens, kwd_labels, batch_size=hparams.BATCH_SIZE, shuffle=False),
                 total=n_batches,  desc="BATCH: "):
        if hparams.USE_CUDA:
            input_seqs_batch = torch.LongTensor(input_seqs_batch).cuda().transpose(0, 1)
        else:
            input_seqs_batch = torch.LongTensor(input_seqs_batch).transpose(0, 1)

        logits = kwd_predictor(input_seqs_batch, input_lens_batch)
        probs = torch.sigmoid(logits).cpu().detach().numpy()
        for prob in probs:
            top_kwd_ids = np.argsort(prob)[::-1][:hparams.SHOW_TOP_KWD]
            top_prob = prob[top_kwd_ids]
            out_kwds.write("\t".join(f"{index2kwd[i]}\t{prob0:.2%}" for (i, prob0) in zip(top_kwd_ids, top_prob))+"\n")
    out_kwds.close()

def cluster2kwd_masks(logits, edge_cnt, index2kwd, kwd2index):
    logits_np = logits.detach().cpu().numpy()
    kwd_masks = [np.zeros_like(logits_np) for i in range(hparams.KWD_CLUSTERS)]
    for record_id, logits_one in enumerate(logits_np):
        if hparams.THRESHOLD < 0:
            top_kwd_ids = np.argsort(logits_one)[::-1][:hparams.SAMPLE_TOP_K]
        else:
            probs_one = scipy.special.softmax(logits_one)
            top_kwd_ids = [i for i, prob in enumerate(probs_one) if prob > hparams.THRESHOLD]
        num_kwds = len(top_kwd_ids)
        if num_kwds > hparams.KWD_CLUSTERS:
            adj_mat = np.zeros((num_kwds, num_kwds))
            for a, b in combinations(range(num_kwds), 2):
                adj_mat[a, b] = edge_cnt[top_kwd_ids[a], top_kwd_ids[b]]
            adj_mat += adj_mat.T
            sc = SpectralClustering(hparams.KWD_CLUSTERS, affinity='precomputed',
                                    assign_labels='discretize')
            pred_groups = sc.fit_predict(adj_mat)
            kwds_groups = [[] for i in range(hparams.KWD_CLUSTERS)]
            group_likelihood = [1 for i in range(hparams.KWD_CLUSTERS)]
            for kwd, pred_group in zip(top_kwd_ids, pred_groups):
                kwds_groups[pred_group].append(kwd)
                kwd_prob = logits_one[kwd]
                group_likelihood[pred_group] = max(group_likelihood[pred_group], kwd_prob)
            priority_group = np.argsort(group_likelihood)[::-1]
            for priority, group_id in enumerate(priority_group):
                # if len(kwds_groups[group_id]) == 0:  # fillna with the most likely group
                #     for group_id2 in priority_group:
                #         if group_id2 == group_id or len(kwds_groups[group_id2]) == 0:
                #             continue
                #         kwds_groups[group_id] = kwds_groups[group_id2][:]
                #         break
                for kwd in kwds_groups[group_id]:
                    kwd_masks[priority][record_id, kwd] = 1
        else:
            for priority, kwd in enumerate(top_kwd_ids):
                kwd_masks[priority][record_id, kwd] = 1
    return kwd_masks


def get_cluster_kwds(kwd_predictor, test_data, edge_cnt, index2kwd, kwd2index):
    id_seqs, input_seqs, input_lens, output_seqs, output_lens, kwd_labels, kwd_filters = test_data
    kwd_predictor.eval()
    n_batches = len(input_seqs) // hparams.BATCH_SIZE
    kwd_masks = [[] for i in range(hparams.KWD_CLUSTERS)]
    for id_seqs_batch, input_seqs_batch, input_lens_batch, kwd_labels_batch, kwd_filters_batch in \
            tqdm(iterate_minibatches(id_seqs, input_seqs, input_lens, kwd_labels, kwd_filters, batch_size=hparams.BATCH_SIZE,
                                     shuffle=False),
                 total=n_batches, desc="CLUSTER: "):
        if hparams.USE_CUDA:
            input_seqs_batch = torch.LongTensor(input_seqs_batch).cuda().transpose(0, 1)
            if hparams.USER_FILTER:
                kwd_filters_batch = torch.FloatTensor(kwd_filters_batch).cuda()
        else:
            input_seqs_batch = torch.LongTensor(input_seqs_batch).transpose(0, 1)
            if hparams.USER_FILTER:
                kwd_filters_batch = torch.FloatTensor(kwd_filters_batch)

        logits = kwd_predictor(input_seqs_batch, input_lens_batch)
        if hparams.USER_FILTER:
            logits += kwd_filters_batch
        kwd_masks_batch = cluster2kwd_masks(logits, edge_cnt, index2kwd, kwd2index)
        for group_id in range(hparams.KWD_CLUSTERS):
            kwd_masks[group_id].append(kwd_masks_batch[group_id])
    for group_id in range(hparams.KWD_CLUSTERS):
        kwd_masks[group_id] = np.concatenate(kwd_masks[group_id], axis=0)
    return kwd_masks

