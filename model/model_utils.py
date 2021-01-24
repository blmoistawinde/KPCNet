import torch
import torch.nn as nn
from torch.nn import functional as F
from constants import *
from hparams import hparams
import copy


def masked_cross_entropy(logits, target, length, loss_fn, mixer_delta=None):
    batch_size = logits.shape[0]
    # log_probs: (batch, max_len, num_classes)
    log_probs = F.log_softmax(logits, dim=2)
    loss = 0.
    for b in range(batch_size):
        curr_len = min(length[b], mixer_delta)
        sent_loss = loss_fn(log_probs[b][:curr_len], target[b][:curr_len]) / curr_len
        loss += sent_loss
    loss = loss / batch_size
    return loss

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def top_k_top_p_sampling(logits, num_samples=5, top_k=20, top_p=0.9, filter_value=-float("Inf"), min_tokens_to_keep=1):
    # Top-p/top-k filtering
    # the function will change the input tensor, must copy!
    filtered_logits = top_k_top_p_filtering(logits.clone().detach(), top_k=top_k, top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)
    samples_bow = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples)
    # [batch_size, num_kwds], with sampled kwd's = 1, otherwise 0
    samples_mat = torch.zeros_like(logits).scatter_(1, samples_bow, 1)
    return samples_mat

def produce_kwd_mask(logits, threshold=-1, num_samples=0):
    # deterministicly select kwds
    if threshold > 0:
        kwd_mask = (torch.sigmoid(logits) >= threshold).float()
    else:
        # randomly select a few kwds with sampling
        kwd_mask = top_k_top_p_sampling(logits, num_samples, hparams.SAMPLE_TOP_K, hparams.SAMPLE_TOP_P,
                                        min_tokens_to_keep=num_samples)
    return kwd_mask

def seq2bow(seq, vocab_size):
    """

    :param seq: [batch_size, seq_len], value in [0, vocab_size]
    :return:  [batch_size, vocab_size]
    """
    batch_size = seq.shape[0]
    bow = torch.zeros((batch_size, vocab_size))
    bow.scatter_add_(1, seq.long(), torch.ones_like(seq, dtype=torch.float))
    return bow

def hamming_diversity(backtrack_seqs, vocab_size):
    """

    :param backtrack_seqs: a group of seqs (BATCH_SIZE, GROUP_BEAM_SIZE, curr_t)
    :param vocab_size:
    :return: penalty_term (BATCH_SIZE, vocab_size) as decoder_out_log_probs
    """
    prev_seqs = backtrack_seqs.detach().clone().view(backtrack_seqs.shape[0], -1)
    penalty_term = -seq2bow(prev_seqs, vocab_size) * hparams.DIVERSE_LAMBDA
    if hparams.USE_CUDA:
        penalty_term = penalty_term.cuda()
    return penalty_term
