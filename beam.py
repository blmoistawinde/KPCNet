import os
from constants import *
from hparams import hparams
from utils import *
from process_data import *
import torch
from model.model_utils import *
from tqdm import tqdm
import pdb
from itertools import combinations

def write_kwd_records(kwd_masks, logits, index2kwd, out_kwds, out_kwd_sp, i=0):
    kwd_masks = kwd_masks.cpu().detach().numpy()
    if out_kwd_sp:
        for kwd_mask in kwd_masks:
            kwd_ids = np.where(kwd_mask != 0)[0]
            out_kwd_sp.write("\t".join(f"{index2kwd[i]}" for i in kwd_ids) + "\n")
    if i == 0:
        probs = torch.sigmoid(logits).cpu().detach().numpy()
        for prob in probs:
            top_kwd_ids = np.argsort(prob)[::-1][:hparams.SHOW_TOP_KWD]
            top_prob = prob[top_kwd_ids]
            out_kwds.write("\t".join(f"{index2kwd[i]}\t{prob0:.2%}" for (i, prob0) in zip(top_kwd_ids, top_prob)) + "\n")

def write_seqs(backtrack_seqs, id_seqs_batch, word2index, index2word, has_ids, out_files, out_ids_files,
               i=None, save_all_beam=True):
    for b in range(hparams.BATCH_SIZE):
        for k in range(hparams.BEAM_SIZE):
            decoded_words = []
            for t in range(backtrack_seqs.shape[2]):
                idx = int(backtrack_seqs[b, k, t])
                if idx == word2index[EOS_token]:
                    decoded_words.append(EOS_token)
                    break
                else:
                    decoded_words.append(index2word[idx])
            if not save_all_beam and k > 0: break  # we only use the first one
            if i is None:
                out_files[k].write(' '.join(decoded_words) + '\n')
            else:
                out_files[k][i].write(' '.join(decoded_words) + '\n')
            if has_ids and (i == 0 or i is None):
                out_ids_files[k].write(id_seqs_batch[b] + '\n')

def produce_seqs(backtrack_seqs, word2index, index2word):
    """
    out_seqs: str list of size hparams.BEAM_SIZE
    """
    assert hparams.BATCH_SIZE == 1
    out_seqs = []
    b = 0
    for k in range(hparams.BEAM_SIZE):
        decoded_words = []
        for t in range(backtrack_seqs.shape[2]):
            idx = int(backtrack_seqs[b, k, t])
            if idx == word2index[EOS_token]:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(index2word[idx])
        out_seqs.append(' '.join(decoded_words) + '\n')
    return out_seqs

def evaluate_beam(word2index, index2word, encoder, decoder, kwd_predictor, kwd_bridge, test_data,
                  max_out_len, out_dir, out_prefix, index2kwd, save_all_beam=True, infer=False):
    
    id_seqs, input_seqs, input_lens, output_seqs, output_lens, kwd_labels, kwd_filters = test_data
    n_batches = len(input_seqs) // hparams.BATCH_SIZE

    encoder.eval()
    decoder.eval()
    kwd_predictor.eval()
    kwd_bridge.eval()
    has_ids = True
    if infer:  # produce str list instead of write into file
        assert hparams.BATCH_SIZE == 1
        out_seqs = []
    else:
        # output for different beams
        out_dir_prefix = os.path.join(out_dir, out_prefix)
        out_kwd_sp = open(out_dir_prefix+".kwd_samples", "w")
        out_kwds = open(out_dir_prefix+".kwds", "w")
        out_files = [None] * hparams.BEAM_SIZE
        out_ids_files = [None] * hparams.BEAM_SIZE
        for k in range(hparams.BEAM_SIZE):
            if not save_all_beam and k > 0: break          # we only use the first one
            out_files[k] = open(out_dir_prefix+'.beam%d' % k, 'w')
            if id_seqs[0] is not None:
                out_ids_files[k] = open(out_dir_prefix+'.beam%d.ids' % k, 'w')
            else:
                has_ids = False

    for id_seqs_batch, input_seqs_batch, input_lens_batch, kwd_labels_batch, kwd_filters_batch in \
            tqdm(iterate_minibatches(id_seqs, input_seqs, input_lens, kwd_labels, kwd_filters,
                                     batch_size=hparams.BATCH_SIZE, shuffle=False),
                 total=n_batches, desc="BATCH: "):

        if hparams.USE_CUDA:
            input_seqs_batch = torch.LongTensor(input_seqs_batch).cuda().transpose(0, 1)
            if hparams.DECODE_USE_KWD_LABEL:
                kwd_labels_batch = torch.FloatTensor(kwd_labels_batch).cuda()
            if hparams.USER_FILTER:
                kwd_filters_batch = torch.FloatTensor(kwd_filters_batch).cuda()
        else:
            input_seqs_batch = torch.LongTensor(input_seqs_batch).transpose(0, 1)
            if hparams.DECODE_USE_KWD_LABEL:
                kwd_labels_batch = torch.FloatTensor(kwd_labels_batch)
            if hparams.USER_FILTER:
                kwd_filters_batch = torch.FloatTensor(kwd_filters_batch)

        # encoder_outputs: (seq_len, batch, hidden_size)
        # encoder_hidden: (num_layers * num_directions, batch, hidden_size)
        encoder_outputs, encoder_hidden = encoder(input_seqs_batch, input_lens_batch, None)
        decoder_input = torch.LongTensor([word2index[SOS_token]] * hparams.BATCH_SIZE)
        if hparams.USE_CUDA:
            decoder_input = decoder_input.cuda()
        # combine 2 directions: (num_layers, batch, hidden_size)
        decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]
        logits = kwd_predictor(input_seqs_batch, input_lens_batch)
        if hparams.USER_FILTER:
            logits += kwd_filters_batch
        if hparams.DECODE_USE_KWD_LABEL:
            e_features, d_features = kwd_bridge(logits, kwd_mask=kwd_labels_batch)
            kwd_masks = kwd_labels_batch
        else:
            kwd_masks = produce_kwd_mask(logits, hparams.THRESHOLD, hparams.SAMPLE_KWD)
            e_features, d_features = kwd_bridge(logits, kwd_masks)

        if not hparams.NO_ENCODER_BRIDGE:
            # Replace SOS token embedding with the features obtained from kwd predictor
            encoder_outputs[0, :, :] = e_features
        # Run through decoder one time step at a time
        if (not hparams.NO_DECODER_BRIDGE) and ((hparams.DECODER_CONDITION_TYPE == 'replace') or
                hparams.DECODER_CONDITION_TYPE == 'concat'):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs, d_features)
        else:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        # decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs, d_features)
        for token in [PAD_token, EOP_token, SOS_token]:
            decoder_output[:, word2index[token]] = -10e20  # special tokens shouldn't appear here
        decoder_out_log_probs = torch.nn.functional.log_softmax(decoder_output, dim=1)
        if not infer:
            write_kwd_records(kwd_masks, logits, index2kwd, out_kwds, out_kwd_sp)

        ## BEAM search below
        n_paths = hparams.BATCH_SIZE * hparams.BEAM_SIZE
        forbidden_tokens = [dict() for _ in range(n_paths)]
        window_repeat_tokens = [set() for _ in range(n_paths)]

        # extract first BEAM_SIZE words, [BATCH_SIZE, BEAM_SIZE]
        log_probs, indices = decoder_out_log_probs.data.topk(hparams.BEAM_SIZE)
        prev_decoder_hiddens = [decoder_hidden] * hparams.BEAM_SIZE   # will differ later, condition on different beams
        prev_backtrack_seqs = torch.zeros(hparams.BATCH_SIZE, hparams.BEAM_SIZE, 1)
        for k in range(hparams.BEAM_SIZE):
            prev_backtrack_seqs[:, k, 0] = indices[:, k]
        prev_seqs_np = prev_backtrack_seqs.cpu().detach().numpy().astype(int)
        backtrack_seqs = None
        for t in range(1, max_out_len):
            beam_vocab_log_probs = None
            beam_vocab_idx = None
            decoder_hiddens = [None] * hparams.BEAM_SIZE
            for k in range(hparams.BEAM_SIZE):
                decoder_input = indices[:, k]
                # decoder_output, decoder_hiddens[k] = decoder(decoder_input, prev_decoder_hiddens[k], encoder_outputs)
                if not hparams.NO_DECODER_BRIDGE and hparams.DECODER_CONDITION_TYPE == 'concat':
                    decoder_output, decoder_hiddens[k] = decoder(decoder_input, prev_decoder_hiddens[k], encoder_outputs, d_features)
                else:
                    decoder_output, decoder_hiddens[k] = decoder(decoder_input, prev_decoder_hiddens[k], encoder_outputs)
                for token in [PAD_token, EOP_token, SOS_token]:
                    decoder_output[:, word2index[token]] = -10e20  # special tokens shouldn't appear here
                decoder_out_log_probs = torch.nn.functional.log_softmax(decoder_output, dim=1)
                
                vocab_log_probs = torch.ones((hparams.BATCH_SIZE, decoder.output_size)) * -float('inf')
                if hparams.USE_CUDA:
                    vocab_log_probs = vocab_log_probs.cuda()
                # make sure EOS has no children
                for b in range(hparams.BATCH_SIZE):
                    if word2index[EOS_token] in prev_backtrack_seqs[b, k, :t]:  # already ended
                        # vocab_log_probs[b] = log_probs[b, k] + decoder_out_log_probs[b][word2index[PAD_token]]
                        vocab_log_probs[b][word2index[PAD_token]] = log_probs[b, k]
                        # print("end", b)
                    else:
                        # vocab_log_probs[b] = log_probs[b, k] + decoder_out_log_probs[b]
                        vocab_log_probs[b] = (log_probs[b, k]*t + decoder_out_log_probs[b])/(t+1)
                        # vocab_log_probs[b] = (log_probs[b, k] * pow(5+t, 0.7) / pow(5+1, 0.7) +
                        #                       decoder_out_log_probs[b]) * pow(5+1, 0.7) / pow(5+t+1, 0.7)
                    if hparams.BLOCK_NGRAM_REPEAT > 0 and t >= hparams.BLOCK_NGRAM_REPEAT:
                        path_idx = b * hparams.BEAM_SIZE + k
                        prev_ngram = tuple(prev_seqs_np[b, k, -(hparams.BLOCK_NGRAM_REPEAT - 1):])
                        block_tokens = forbidden_tokens[path_idx].get(prev_ngram, None)
                        if block_tokens is not None:
                            vocab_log_probs[b][list(block_tokens - {0,1,2,3})] = -10e20
                    
                    if hparams.AVOID_REPEAT_WINDOW > 0:
                        path_idx = b * hparams.BEAM_SIZE + k
                        block_tokens = list(window_repeat_tokens[path_idx] - {0,1,2,3})
                        if len(block_tokens) > 0:
                            vocab_log_probs[b][block_tokens] = -10e20

                topv, topi = vocab_log_probs.data.topk(decoder.output_size)
                if k == 0:
                    beam_vocab_log_probs = topv
                    beam_vocab_idx = topi
                else:
                    # stack the current beam outputs(output_size) condition on(*) prev beams (BEAM_SIZE)
                    beam_vocab_log_probs = torch.cat((beam_vocab_log_probs, topv), dim=1)
                    beam_vocab_idx = torch.cat((beam_vocab_idx, topi), dim=1)
            # beam_vocab_log_probs: [batch_size, BEAM_SIZE*decoder.output_size]
            topv, topi = beam_vocab_log_probs.data.topk(hparams.BEAM_SIZE)
            indices = torch.zeros(hparams.BATCH_SIZE, hparams.BEAM_SIZE, dtype=torch.long)
            prev_decoder_hiddens = torch.zeros(hparams.BEAM_SIZE, decoder_hiddens[0].shape[0],
                                               hparams.BATCH_SIZE, decoder_hiddens[0].shape[2])
            if hparams.USE_CUDA:
                indices = indices.cuda()
                prev_decoder_hiddens = prev_decoder_hiddens.cuda()
            backtrack_seqs = torch.zeros(hparams.BATCH_SIZE, hparams.BEAM_SIZE, t + 1)
            # get output for the current timestep
            all_ended = True
            for b in range(hparams.BATCH_SIZE):
                indices[b] = torch.index_select(beam_vocab_idx[b], 0, topi[b])
                backtrack_seqs[b, :, t] = indices[b]
                for k in range(hparams.BEAM_SIZE):
                    if word2index[EOS_token] in prev_backtrack_seqs[b, topi[b][k]//decoder.output_size, :t]:
                        backtrack_seqs[b, k, :t] = prev_backtrack_seqs[b, topi[b][k]//decoder.output_size, :t]
                        continue
                    all_ended = False
                    # topi[b][k]//decoder.output_size to get the corresponding prev BEAM
                    prev_decoder_hiddens[k, :, b, :] = decoder_hiddens[topi[b][k]//decoder.output_size][:, b, :]
                    backtrack_seqs[b, k, :t] = prev_backtrack_seqs[b, topi[b][k]//decoder.output_size, :t]
            if all_ended:
                break

            prev_backtrack_seqs = backtrack_seqs
            prev_seqs_np = prev_backtrack_seqs.cpu().detach().numpy().astype(int)
            # update block
            if hparams.BLOCK_NGRAM_REPEAT > 0 and t >= hparams.BLOCK_NGRAM_REPEAT - 1:
                # import pdb; pdb.set_trace()
                for b in range(hparams.BATCH_SIZE):
                    for k in range(hparams.BEAM_SIZE):
                        path_idx = b * hparams.BEAM_SIZE + k
                        curr_ngram = tuple(prev_seqs_np[b, k, -hparams.BLOCK_NGRAM_REPEAT:])
                        forbidden_tokens[path_idx].setdefault(curr_ngram[:-1], set())
                        forbidden_tokens[path_idx][curr_ngram[:-1]].add(curr_ngram[-1])
            # avoid 1-gram repeat with the previous 2
            if hparams.AVOID_REPEAT_WINDOW > 0:
                for b in range(hparams.BATCH_SIZE):
                    for k in range(hparams.BEAM_SIZE):
                        path_idx = b * hparams.BEAM_SIZE + k
                        prev_tokens = set(prev_seqs_np[b, k, -hparams.AVOID_REPEAT_WINDOW:])
                        window_repeat_tokens[path_idx] = prev_tokens
            log_probs = topv
        if not infer:
            write_seqs(backtrack_seqs, id_seqs_batch, word2index, index2word, has_ids, out_files, out_ids_files, save_all_beam=save_all_beam)
        else:
            out_seqs.extend(produce_seqs(backtrack_seqs, word2index, index2word))

    if not infer:
        out_kwd_sp.close()
        out_kwds.close()
        for k in range(hparams.BEAM_SIZE):
            if not save_all_beam and k > 0: break          # we only use the first one
            out_files[k].close()
        print('Beam search Done')
    else:
        return out_seqs


def evaluate_sample(word2index, index2word, encoder, decoder, kwd_predictor, kwd_bridge, test_data,
                  max_out_len, out_dir, out_prefix, index2kwd, sample_times=6):
    id_seqs, input_seqs, input_lens, output_seqs, output_lens, kwd_labels, kwd_filters = test_data
    n_batches = len(input_seqs) // hparams.BATCH_SIZE

    encoder.eval()
    decoder.eval()
    kwd_predictor.eval()
    kwd_bridge.eval()
    has_ids = True
    # output for different beams
    out_dir_prefix = os.path.join(out_dir, out_prefix)
    out_kwd_sp = open(out_dir_prefix + ".kwd_samples", "w")
    out_kwds = open(out_dir_prefix + ".kwds", "w")
    out_files = [open(out_dir_prefix + ".beam%d" % i, "w") for i in range(sample_times)]

    for id_seqs_batch, input_seqs_batch, input_lens_batch, kwd_labels_batch, kwd_filters_batch in \
            tqdm(iterate_minibatches(id_seqs, input_seqs, input_lens, kwd_labels, kwd_filters,
                                     batch_size=hparams.BATCH_SIZE, shuffle=False),
                 total=n_batches, desc="BATCH: "):

        if hparams.USE_CUDA:
            input_seqs_batch = torch.LongTensor(input_seqs_batch).cuda().transpose(0, 1)
            if hparams.DECODE_USE_KWD_LABEL:
                kwd_labels_batch = torch.FloatTensor(kwd_labels_batch).cuda()
            if hparams.USER_FILTER:
                kwd_filters_batch = torch.FloatTensor(kwd_filters_batch).cuda()
        else:
            input_seqs_batch = torch.LongTensor(input_seqs_batch).transpose(0, 1)
            if hparams.DECODE_USE_KWD_LABEL:
                kwd_labels_batch = torch.FloatTensor(kwd_labels_batch)
            if hparams.USER_FILTER:
                kwd_filters_batch = torch.FloatTensor(kwd_filters_batch)

        encoder_outputs, encoder_hidden = encoder(input_seqs_batch, input_lens_batch, None)
        # decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]
        logits = kwd_predictor(input_seqs_batch, input_lens_batch)
        if hparams.USER_FILTER:
            logits += kwd_filters_batch
        if hparams.DECODE_USE_KWD_LABEL:
            e_features, d_features = kwd_bridge(logits, kwd_mask=kwd_labels_batch)
            kwd_masks = kwd_labels_batch
        else:
            kwd_masks = produce_kwd_mask(logits, hparams.THRESHOLD, hparams.SAMPLE_KWD)
            e_features, d_features = kwd_bridge(logits, kwd_masks)

        write_kwd_records(kwd_masks, logits, index2kwd, out_kwds, out_kwd_sp)

        if not hparams.NO_ENCODER_BRIDGE:
            # Replace SOS token embedding with the features obtained from kwd predictor
            encoder_outputs[0, :, :] = e_features
        for sp in range(sample_times):
            decoder_input = torch.LongTensor([word2index[SOS_token]] * hparams.BATCH_SIZE)
            if hparams.USE_CUDA:
                decoder_input = decoder_input.cuda()
            decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]
            seqs = np.zeros((hparams.BATCH_SIZE, max_out_len), dtype=int)
            for t in range(max_out_len):
                if (not hparams.NO_DECODER_BRIDGE) and ((t == 0 and hparams.DECODER_CONDITION_TYPE == 'replace') or
                        hparams.DECODER_CONDITION_TYPE == 'concat'):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs, d_features)
                else:
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                for token in [PAD_token, EOP_token, SOS_token]:
                    decoder_output[:, word2index[token]] = -10e20  # special tokens shouldn't appear here
                # top-K top-p sampling
                filtered_logits = top_k_top_p_filtering(decoder_output.clone().detach(),
                                                        top_k=20, top_p=0.9)
                # sample top one: [batch_size, 1]
                decoder_input = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1).squeeze(1)
                seqs[:, t] = decoder_input.detach().cpu().numpy()

            for b in range(hparams.BATCH_SIZE):
                decoded_words = []
                for t in range(max_out_len):
                    idx = int(seqs[b, t])
                    if idx == word2index[EOS_token]:
                        decoded_words.append(EOS_token)
                        break
                    else:
                        decoded_words.append(index2word[idx])
                out_files[sp].write(' '.join(decoded_words) + '\n')
    out_kwd_sp.close()
    out_kwds.close()
    [f.close() for f in out_files]


# slightly different implement with the original one, each group have same time
def evaluate_diverse_beam(word2index, index2word, encoder, decoder, kwd_predictor, kwd_bridge, test_data,
                  max_out_len, out_dir, out_prefix, index2kwd, save_all_beam=True):
    id_seqs, input_seqs, input_lens, output_seqs, output_lens, kwd_labels, kwd_filters = test_data
    n_batches = len(input_seqs) // hparams.BATCH_SIZE
    beam_each = hparams.BEAM_SIZE // hparams.DIVERSE_GROUP
    encoder.eval()
    decoder.eval()
    kwd_predictor.eval()
    kwd_bridge.eval()
    has_ids = True
    # output for different beams
    out_dir_prefix = os.path.join(out_dir, out_prefix)
    out_kwd_sp = open(out_dir_prefix + ".kwd_samples", "w")
    out_kwds = open(out_dir_prefix + ".kwds", "w")
    out_files = [None] * hparams.BEAM_SIZE
    out_ids_files = [None] * hparams.BEAM_SIZE
    for k in range(hparams.BEAM_SIZE):
        if not save_all_beam and k > 0: break  # we only use the first one
        out_files[k] = open(out_dir_prefix + '.beam%d' % k, 'w')
        if id_seqs[0] is not None:
            out_ids_files[k] = open(out_dir_prefix + '.beam%d.ids' % k, 'w')
        else:
            has_ids = False

    for id_seqs_batch, input_seqs_batch, input_lens_batch, kwd_labels_batch, kwd_filters_batch in \
            tqdm(iterate_minibatches(id_seqs, input_seqs, input_lens, kwd_labels, kwd_filters,
                                     batch_size=hparams.BATCH_SIZE, shuffle=False),
                 total=n_batches, desc="BATCH: "):

        if hparams.USE_CUDA:
            input_seqs_batch = torch.LongTensor(input_seqs_batch).cuda().transpose(0, 1)
            if hparams.DECODE_USE_KWD_LABEL:
                kwd_labels_batch = torch.FloatTensor(kwd_labels_batch).cuda()
            if hparams.USER_FILTER:
                kwd_filters_batch = torch.FloatTensor(kwd_filters_batch).cuda()
        else:
            input_seqs_batch = torch.LongTensor(input_seqs_batch).transpose(0, 1)
            if hparams.DECODE_USE_KWD_LABEL:
                kwd_labels_batch = torch.FloatTensor(kwd_labels_batch)
            if hparams.USER_FILTER:
                kwd_filters_batch = torch.FloatTensor(kwd_filters_batch)

        # encoder_outputs: (seq_len, batch, hidden_size)
        # encoder_hidden: (num_layers * num_directions, batch, hidden_size)
        encoder_outputs, encoder_hidden = encoder(input_seqs_batch, input_lens_batch, None)
        decoder_input = torch.LongTensor([word2index[SOS_token]] * hparams.BATCH_SIZE)
        if hparams.USE_CUDA:
            decoder_input = decoder_input.cuda()
        # combine 2 directions: (num_layers, batch, hidden_size)
        decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]
        logits = kwd_predictor(input_seqs_batch, input_lens_batch)
        if hparams.USER_FILTER:
            logits += kwd_filters_batch
        if hparams.DECODE_USE_KWD_LABEL:
            e_features, d_features = kwd_bridge(logits, kwd_mask=kwd_labels_batch)
            kwd_masks = kwd_labels_batch
        else:
            kwd_masks = produce_kwd_mask(logits, hparams.THRESHOLD, hparams.SAMPLE_KWD)
            e_features, d_features = kwd_bridge(logits, kwd_masks)

        if not hparams.NO_ENCODER_BRIDGE:
            # Replace SOS token embedding with the features obtained from kwd predictor
            encoder_outputs[0, :, :] = e_features
        # Run through decoder one time step at a time
        if (not hparams.NO_DECODER_BRIDGE) and ((hparams.DECODER_CONDITION_TYPE == 'replace') or
                hparams.DECODER_CONDITION_TYPE == 'concat'):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs, d_features)
        else:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        for token in [PAD_token, EOP_token, SOS_token]:
            decoder_output[:, word2index[token]] = -10e20  # special tokens shouldn't appear here
        decoder_out_log_probs = torch.nn.functional.log_softmax(decoder_output, dim=1)

        write_kwd_records(kwd_masks, logits, index2kwd, out_kwds, out_kwd_sp)

        ## BEAM search below
        n_paths = hparams.BATCH_SIZE * hparams.BEAM_SIZE
        forbidden_tokens = [dict() for _ in range(n_paths)]
        window_repeat_tokens = [set() for _ in range(n_paths)]

        # extract first BEAM_SIZE words, [BATCH_SIZE, BEAM_SIZE]
        log_probs, prev_indices = decoder_out_log_probs.data.topk(hparams.BEAM_SIZE)
        prev_decoder_hiddens = torch.stack([decoder_hidden] * hparams.BEAM_SIZE)  # will differ later, condition on different beams
        prev_backtrack_seqs = torch.zeros(hparams.BATCH_SIZE, hparams.BEAM_SIZE, 1)
        for k in range(hparams.BEAM_SIZE):
            prev_backtrack_seqs[:, k, 0] = prev_indices[:, k]
        prev_seqs_np = prev_backtrack_seqs.cpu().detach().numpy().astype(int)
        backtrack_seqs = None
        for t in range(1, max_out_len):
            all_ended = True
            beam_vocab_log_probs = None
            beam_vocab_idx = None
            decoder_hiddens = [None] * hparams.BEAM_SIZE
            backtrack_seqs = torch.zeros(hparams.BATCH_SIZE, hparams.BEAM_SIZE, t + 1)
            for g in range(hparams.DIVERSE_GROUP):
                if g > 0:
                    ### group similarity term for divere beam search
                    sim_term = hamming_diversity(prev_backtrack_seqs[:, :g * beam_each, t:t + 1],
                                                 decoder.output_size)
                for k0 in range(beam_each):
                    k = k0 + g*beam_each
                    decoder_input = prev_indices[:, k]
                    if not hparams.NO_DECODER_BRIDGE and hparams.DECODER_CONDITION_TYPE == 'concat':
                        decoder_output, decoder_hiddens[k] = decoder(decoder_input, prev_decoder_hiddens[k], encoder_outputs,
                                                                     d_features)
                    else:
                        decoder_output, decoder_hiddens[k] = decoder(decoder_input, prev_decoder_hiddens[k], encoder_outputs)
                    # decoder_output, decoder_hiddens[k] = decoder(decoder_input, prev_decoder_hiddens[k], encoder_outputs)
                    for token in [PAD_token, EOP_token, SOS_token]:
                        decoder_output[:, word2index[token]] = -10e20  # special tokens shouldn't appear here
                    decoder_out_log_probs = torch.nn.functional.log_softmax(decoder_output, dim=1)

                    vocab_log_probs = torch.ones((hparams.BATCH_SIZE, decoder.output_size)) * -float('inf')
                    if hparams.USE_CUDA:
                        vocab_log_probs = vocab_log_probs.cuda()

                    # make sure EOS has no children
                    for b in range(hparams.BATCH_SIZE):
                        if word2index[EOS_token] in prev_backtrack_seqs[b, k, :t]:  # already ended
                            # vocab_log_probs[b] = log_probs[b, k] + decoder_out_log_probs[b][word2index[PAD_token]]
                            vocab_log_probs[b][word2index[PAD_token]] = log_probs[b, k]

                        else:
                            vocab_log_probs[b] = (log_probs[b, k]*t + decoder_out_log_probs[b])/(t+1)
                            # vocab_log_probs[b] = (log_probs[b, k] * t + decoder_out_log_probs[b]) / (t + 1)

                        if hparams.BLOCK_NGRAM_REPEAT > 0 and t >= hparams.BLOCK_NGRAM_REPEAT:
                            path_idx = b * hparams.BEAM_SIZE + k
                            prev_ngram = tuple(prev_seqs_np[b, k, -(hparams.BLOCK_NGRAM_REPEAT - 1):])
                            block_tokens = forbidden_tokens[path_idx].get(prev_ngram, None)
                            if block_tokens is not None:
                                vocab_log_probs[b][list(block_tokens - {0,1,2,3})] = -10e20

                        if hparams.AVOID_REPEAT_WINDOW > 0:
                            path_idx = b * hparams.BEAM_SIZE + k
                            block_tokens = list(window_repeat_tokens[path_idx] - {0,1,2,3})
                            if len(block_tokens) > 0:
                                vocab_log_probs[b][block_tokens] = -10e20
                    if g > 0:
                        ### group similarity term for divere beam search
                        vocab_log_probs += sim_term

                    topv, topi = vocab_log_probs.data.topk(decoder.output_size)
                    if k0 == 0:
                        beam_vocab_log_probs = topv
                        beam_vocab_idx = topi
                    else:
                        # stack the current beam outputs(output_size) condition on(*) prev beams (BEAM_SIZE)
                        beam_vocab_log_probs = torch.cat((beam_vocab_log_probs, topv), dim=1)
                        beam_vocab_idx = torch.cat((beam_vocab_idx, topi), dim=1)
                # beam_vocab_log_probs: [batch_size, BEAM_SIZE*decoder.output_size]
                topv, topi = beam_vocab_log_probs.data.topk(beam_each)
                log_probs[:, g*beam_each:(g+1)*beam_each] = topv
                indices = torch.zeros(hparams.BATCH_SIZE, beam_each, dtype=torch.long)
                if hparams.USE_CUDA:
                    indices = indices.cuda()

                for b in range(hparams.BATCH_SIZE):
                    indices[b] = torch.index_select(beam_vocab_idx[b], 0, topi[b])
                    backtrack_seqs[b, g * beam_each:(g + 1) * beam_each, t] = indices[b]
                    prev_indices[b, g * beam_each:(g + 1) * beam_each] = indices[b]
                    for k0 in range(beam_each):
                        k = k0 + g*beam_each
                        # topi[b][k]//decoder.output_size to get the corresponding prev BEAM
                        prev_decoder_hiddens[k, :, b, :] = decoder_hiddens[g*beam_each + topi[b][k0] // decoder.output_size][:, b, :]
                        backtrack_seqs[b, k, :t] = prev_backtrack_seqs[b, g*beam_each + topi[b][k0] // decoder.output_size, :t]
                        if word2index[EOS_token] not in prev_backtrack_seqs[b, g * beam_each + topi[b][k0] // decoder.output_size, :t]:
                            all_ended = False
            if all_ended:
                break

            prev_backtrack_seqs = backtrack_seqs
            prev_seqs_np = prev_backtrack_seqs.cpu().detach().numpy().astype(int)
            # update block
            if hparams.BLOCK_NGRAM_REPEAT > 0 and t >= hparams.BLOCK_NGRAM_REPEAT - 1:
                # import pdb; pdb.set_trace()
                for b in range(hparams.BATCH_SIZE):
                    for k in range(hparams.BEAM_SIZE):
                        path_idx = b * hparams.BEAM_SIZE + k
                        curr_ngram = tuple(prev_seqs_np[b, k, -hparams.BLOCK_NGRAM_REPEAT:])
                        forbidden_tokens[path_idx].setdefault(curr_ngram[:-1], set())
                        forbidden_tokens[path_idx][curr_ngram[:-1]].add(curr_ngram[-1])
            # avoid 1-gram repeat with the previous 2
            if hparams.AVOID_REPEAT_WINDOW > 0:
                for b in range(hparams.BATCH_SIZE):
                    for k in range(hparams.BEAM_SIZE):
                        path_idx = b * hparams.BEAM_SIZE + k
                        prev_tokens = set(prev_seqs_np[b, k, -hparams.AVOID_REPEAT_WINDOW:])
                        window_repeat_tokens[path_idx] = prev_tokens

        write_seqs(backtrack_seqs, id_seqs_batch, word2index, index2word, has_ids, out_files, out_ids_files,
                   save_all_beam=save_all_beam)

    out_kwd_sp.close()
    out_kwds.close()
    for k in range(hparams.BEAM_SIZE):
        if not save_all_beam and k > 0: break  # we only use the first one
        out_files[k].close()
    print('Diverse Beam search Done')
