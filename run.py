import os
from model.attnDecoderRNN import *
from constants import *
from model.encoderRNN import *
from model.kwdPredictor import *
from model.kwdBridge import *
from evaluate import *
from train import *
import torch
import torch.optim as optim
from process_data import *
from tqdm import tqdm
from hparams import hparams
from utils import *
import re

def run_seq2seq(train_data, test_data, word2index, word_embeddings,
                max_target_length, kwd_weight=None, update_kwd_predictor=False,
                train_kwds=None, test_kwds=None, kwd2index=None, kwd_model_dir=None, save_dir="./ckpt", load_models_dir=None):
    print('Initializing models')
    load_kwd_model = (kwd_model_dir is not None) and (load_models_dir is None)
    encoder = EncoderRNN(hparams.HIDDEN_SIZE, word_embeddings, hparams.RNN_LAYERS,
                         dropout=hparams.DROPOUT, update_wd_emb=hparams.UPDATE_WD_EMB)
    decoder = AttnDecoderRNN(hparams.HIDDEN_SIZE, len(word2index), word_embeddings, hparams.ATTN_TYPE,
                             hparams.RNN_LAYERS, dropout=hparams.DROPOUT, update_wd_emb=hparams.UPDATE_WD_EMB,
                             condition=hparams.DECODER_CONDITION_TYPE)
    kwd_predictor = get_predictor(word_embeddings, hparams)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=hparams.LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=hparams.LEARNING_RATE * hparams.DECODER_LEARNING_RATIO)
    if not hparams.WITH_MEMORY:
        kwd_bridge = MLPBridge(hparams.HIDDEN_SIZE, hparams.MAX_KWD, hparams.HIDDEN_SIZE, len(word_embeddings[0]),
                               norm_type=hparams.BRIDGE_NORM_TYPE, dropout=hparams.DROPOUT)
    else:
        kwd_bridge = MemoryBridge(word_embeddings, hparams.MAX_KWD, hparams.HIDDEN_SIZE, len(word_embeddings[0]),
                                  memory_hops=hparams.MEMORY_HOPS)
    kwd_bridge_optimizer = optim.Adam(kwd_bridge.parameters(), lr=hparams.LEARNING_RATE)
    print(kwd_bridge)
    if hparams.USE_CUDA:
        encoder.cuda()
        decoder.cuda()
        kwd_predictor.cuda()
        kwd_bridge.cuda()
        if load_kwd_model:
            kwd_predictor.load_state_dict(torch.load(kwd_model_dir))
    else:
        if load_kwd_model:
            kwd_predictor.load_state_dict(torch.load(kwd_model_dir, map_location='cpu'))

    if load_models_dir is not None:
        if hparams.USE_CUDA:
            models = torch.load(load_models_dir)
        else:
            models = torch.load(load_models_dir, map_location='cpu')
        encoder.load_state_dict(models["encoder"])
        decoder.load_state_dict(models["decoder"])
        kwd_predictor.load_state_dict(models["kwd_predictor"])
        kwd_bridge.load_state_dict(models["kwd_bridge"])
        epoch0 = int(re.search(r"epoch(\d+)\.", load_models_dir).group(1)) + 1
        print(f"Loading model of {epoch0} from {load_models_dir}")
    else:
        epoch0 = 0

    if not update_kwd_predictor:
        assert load_kwd_model or load_models_dir is not None
        for param in kwd_predictor.parameters():
            param.requires_grad = False
        optimizers = [encoder_optimizer, decoder_optimizer, kwd_bridge_optimizer]
    else:
        kwd_pred_optimizer = optim.Adam(kwd_predictor.parameters(), lr=hparams.LEARNING_RATE)
        optimizers = [encoder_optimizer, decoder_optimizer, kwd_bridge_optimizer, kwd_pred_optimizer]

    print_loss_total = 0  # Reset every print_every

    if train_kwds is None and test_kwds is None:
        ids_seqs, input_seqs, input_lens, output_seqs, output_lens, kwd_labels, kwd_masks = train_data
    else:
        ids_seqs, input_seqs, input_lens, output_seqs, output_lens = train_data
    n_batches = len(input_seqs) // hparams.BATCH_SIZE
    teacher_forcing_ratio = 1.0
    if hparams.SCHEDULED_SAMPLE:
        decr = (teacher_forcing_ratio - hparams.MIN_TF_RATIO) / hparams.N_EPOCHS   # linear decay
    else:          # always teacher forcing
        decr = 0

    for epoch in range(epoch0, hparams.N_EPOCHS):
        if train_kwds is not None and kwd2index is not None:
            kwd_labels, kwd_masks = build_kwd_arr(train_kwds, kwd2index)

        for ids_seqs_batch, input_seqs_batch, input_lens_batch, \
            output_seqs_batch, output_lens_batch, kwd_labels_batch, kwd_masks_batch in \
                tqdm(iterate_minibatches(ids_seqs, input_seqs, input_lens, output_seqs, output_lens, kwd_labels, kwd_masks, batch_size=hparams.BATCH_SIZE),
                     total=n_batches, desc=f"EPOCH {epoch}: "):
            loss = train(
                input_seqs_batch, input_lens_batch,
                output_seqs_batch, output_lens_batch, kwd_labels_batch, kwd_masks_batch,
                encoder, decoder, kwd_predictor, kwd_bridge,
                optimizers, word2index[SOS_token], max_target_length,
                hparams.BATCH_SIZE, teacher_forcing_ratio, kwd_weight
            )
            print_loss_total += loss
    
        teacher_forcing_ratio = teacher_forcing_ratio - decr
        print_loss_avg = print_loss_total / n_batches
        print_loss_total = 0
        print('Epoch: %d' % epoch)
        print('Train Loss: %.5f' % (print_loss_avg))
        curr_test_loss = evaluate(test_data, encoder, decoder, kwd_predictor, kwd_bridge, word2index[SOS_token],
                                  max_target_length, hparams.BATCH_SIZE, kwd_weight, test_kwds, kwd2index)
        print('Dev Loss: %.4f ' % curr_test_loss)

        if epoch == 0 or (epoch + 1) % hparams.SAVE_EPOCH_INTERVAL == 0:
            if kwd_model_dir:
                kwd_model_name = kwd_model_dir[kwd_model_dir.rfind("/")+1:kwd_model_dir.rfind(".")]
                kwd_model_prefix = kwd_model_name[len("kwd"):-len(".best")]
                model_name = hparams.get_exp_name(kwd_model_prefix) + ".epoch%d.models" % epoch
            elif load_models_dir:
                model_name = load_models_dir[load_models_dir.rfind("/")+1:load_models_dir.rfind(".epoch")] + ".epoch%d.models" % epoch
            else:
                model_name = hparams.get_exp_name("model") + ".epoch%d.models" % epoch
            print('Saving model params')
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "kwd_predictor": kwd_predictor.state_dict(),
                "kwd_bridge": kwd_bridge.state_dict()
            }, os.path.join(save_dir, model_name))

def run_kwd(train_data, test_data, index2kwd, word_embeddings,
            kwd_weight=None, train_kwds=None, test_kwds=None, kwd2index=None, save_dir="./ckpt"):
    # Initialize q models
    print('Initializing models')
    kwd_predictor = get_predictor(word_embeddings, hparams)
    kwd_optimizer = optim.Adam(kwd_predictor.parameters(), lr=hparams.LEARNING_RATE)

    # Move models to GPU
    if hparams.USE_CUDA:
        kwd_predictor.cuda()
    if train_kwds is None and test_kwds is None:
        ids_seqs, input_seqs, input_lens, output_seqs, output_lens, kwd_labels, kwd_masks = train_data
    else:
        ids_seqs, input_seqs, input_lens, output_seqs, output_lens = train_data

    
    n_batches = len(input_seqs) // hparams.BATCH_SIZE
    num_decrease = 0
    best_epoch, best_test_loss = 0, float("inf")
    epoch0 = 0
    print_loss_total = 0  # Reset every epoch
    for epoch in range(epoch0, hparams.N_EPOCHS):
        if train_kwds is not None and kwd2index is not None:
            kwd_labels, kwd_masks = build_kwd_arr(train_kwds, kwd2index)
        
        for ids_seqs_batch, input_seqs_batch, input_lens_batch, \
            output_seqs_batch, output_lens_batch, kwd_labels_batch, kwd_masks_batch in \
                tqdm(iterate_minibatches(ids_seqs, input_seqs, input_lens, output_seqs, output_lens,
                                         kwd_labels, kwd_masks, batch_size=hparams.BATCH_SIZE),
                     total=n_batches, desc="BATCH: "):

            # Run the train function
            loss = train_kwd(
                input_seqs_batch, input_lens_batch,
                kwd_labels_batch, kwd_masks_batch,
                kwd_predictor, kwd_optimizer, kwd_weight
            )
    
            # Keep track of loss
            print_loss_total += loss
    
        # teacher_forcing_ratio = teacher_forcing_ratio - decr
        print_loss_avg = print_loss_total / n_batches
        print_loss_total = 0
        print('Epoch: %d' % epoch)
        print('Train Loss: %.7f' % (print_loss_avg))

        # out_fname = hparams.get_exp_name()+".epoch%d.kwd_prob" % epoch \
        #     if epoch == 0 or (epoch + 1) % hparams.SAVE_EPOCH_INTERVAL == 0 else None
        # out_fname = os.path.join(save_dir, out_fname) if out_fname is not None else None
        out_fname = None
        curr_test_loss = evaluate_kwd(index2kwd, kwd_predictor, test_data, out_fname,
                                      kwd_weight, test_kwds, kwd2index)
        print('Dev Loss: %.7f ' % curr_test_loss)
        # can use early stopping here
        if curr_test_loss >= best_test_loss:
            num_decrease += 1
        else:
            best_epoch, best_test_loss = epoch, curr_test_loss
            num_decrease = 0
            # use the same name all the time, can overwrite
            print('Saving best model params')
            torch.save(kwd_predictor.state_dict(), os.path.join(save_dir, hparams.get_exp_name()+".best.kwd_pred"))
        if num_decrease >= hparams.PATIENCE:
            print('Early stopping, save last model')
            print('Find best at epoch %d' % best_epoch)
            torch.save(kwd_predictor.state_dict(), os.path.join(save_dir, hparams.get_exp_name()+".last.kwd_pred"))
            return
    print('Find best at epoch %d' % best_epoch)
    torch.save(kwd_predictor.state_dict(), os.path.join(save_dir, hparams.get_exp_name() + ".last.kwd_pred"))
    return
