from constants import *
from model.model_utils import *
import numpy as np
import random
import torch
from torch.autograd import Variable
from hparams import hparams
from utils import *


def train(input_batches, input_lens, target_batches, target_lens, kwd_labels, kwd_masks,
          encoder, decoder, kwd_predictor, kwd_bridge, optimizers,
          SOS_idx, max_target_length, batch_size, teacher_forcing_ratio, kwd_weight=None):
    # if not update_kwd_predictor, kwd_predictor_optimizer will not be included
    for optimizer0 in optimizers:
        optimizer0.zero_grad()

    if hparams.USE_CUDA:
        input_batches = torch.LongTensor(input_batches).cuda().transpose(0, 1)
        target_batches = torch.LongTensor(target_batches).cuda().transpose(0, 1)
        kwd_labels = torch.FloatTensor(kwd_labels).cuda()
        kwd_masks = torch.FloatTensor(kwd_masks).cuda()
    else:
        input_batches = torch.LongTensor(input_batches).transpose(0, 1)
        target_batches = torch.LongTensor(target_batches).transpose(0, 1)
        kwd_labels = torch.FloatTensor(kwd_labels)
        kwd_masks = torch.FloatTensor(kwd_masks)

    # Run post words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lens, None)
    logits = kwd_predictor(input_batches, input_lens)
    e_features, d_features = kwd_bridge(logits, kwd_mask=kwd_labels)            # use label as mask in training
    # Prepare input and output variables [2, batch_size, hidden_size]
    decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]
    if not hparams.NO_ENCODER_BRIDGE:
        ### Replace SOS token embedding with the features obtained from kwd predictor
        encoder_outputs[0, :, :] = e_features
    # masked loss
    if not hparams.FREEZE_KWD_MODEL:
        if kwd_weight is None:
            loss_kwd = torch.nn.BCEWithLogitsLoss()(logits*kwd_masks, kwd_labels)
        else:
            loss_kwd = torch.nn.BCEWithLogitsLoss(pos_weight=kwd_weight)(logits*kwd_masks, kwd_labels)

    if hparams.USE_CUDA:
        decoder_input = torch.LongTensor([SOS_idx] * batch_size).cuda()
        # all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size).cuda()
    else:
        decoder_input = torch.LongTensor([SOS_idx] * batch_size)
        # all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)
    all_decoder_outputs = []

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        if (not hparams.NO_DECODER_BRIDGE) and ((t == 0 and hparams.DECODER_CONDITION_TYPE == 'replace') or \
                hparams.DECODER_CONDITION_TYPE == 'concat'):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs, d_features)
        else:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        # all_decoder_outputs[t] = decoder_output
        all_decoder_outputs.append(decoder_output)
        if use_teacher_forcing:
            decoder_input = target_batches[t]           # Next input is current target
        else:                                           # Greeding decoding
            for b in range(batch_size):
                topi = decoder_output[b].topk(1)[1][0]    
                decoder_input[b] = topi.squeeze().detach()

    all_decoder_outputs = torch.cat(all_decoder_outputs, dim=0).view(max_target_length, batch_size, decoder.output_size)
    loss_fn = torch.nn.NLLLoss()
    loss_seq2seq = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lens, loss_fn, max_target_length
    )

    loss = loss_seq2seq if hparams.FREEZE_KWD_MODEL else loss_seq2seq + hparams.KWD_LOSS_RATIO * loss_kwd

    loss.backward()
    for model in (encoder, decoder, kwd_predictor, kwd_bridge):
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.MAX_GRAD_NORM)
    for optimizer0 in optimizers:
        optimizer0.step()
    return loss.item()

def train_kwd(input_batches, input_lens, kwd_labels, kwd_masks, kwd_predictor, kwd_optimizer, kwd_weight=None):
    # Zero gradients of both optimizers
    kwd_optimizer.zero_grad()
    if hparams.NO_NEG_SAMPLE:
        kwd_masks = torch.ones(kwd_labels.shape)
    if hparams.USE_CUDA:
        input_batches = torch.LongTensor(input_batches).cuda().transpose(0, 1)
        kwd_labels = torch.FloatTensor(kwd_labels).cuda()
        kwd_masks = torch.FloatTensor(kwd_masks).cuda()
    else:
        input_batches = torch.LongTensor(input_batches).transpose(0, 1)
        kwd_labels = torch.FloatTensor(kwd_labels)
        kwd_masks = torch.FloatTensor(kwd_masks)

    logits = kwd_predictor(input_batches, input_lens)
    # masked loss
    if kwd_weight is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=kwd_weight)
    loss = criterion(logits*kwd_masks, kwd_labels)
    if hparams.KWD_PREDICTOR_TYPE == 'cnnpost':
        # additional loss
        loss += criterion(kwd_predictor.logits_pre*kwd_masks, kwd_labels)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(kwd_predictor.parameters(), hparams.MAX_GRAD_NORM)
    kwd_optimizer.step()
    return loss.item()
