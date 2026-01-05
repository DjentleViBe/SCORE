"""Training riffs using ML"""
import os
import platform
import argparse
import random
import sys
import shutil
import csv
import numpy as np
import guitarpro as gp
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import config as cfg
from preprocess import readgpro, guitarinfo, get_positional_encoding, create_dir
from postprocess import plot_multiple, plotbar, plotbarlog, plotbar_dual, \
                makegpro, writegpro, writebincount, writebincount2, readbincount, kldivergence
from encoding import tokenizer_1, note_prob, beat_prob, beattype_prob
from decoding import detokenizer_1
from _decoder.decoder import DecoderAPE
from testing import testing, inference
from _loss.customloss import RepetitionPenaltyLossForSpecificTokens, SequenceMemory
from validation import validation
from fileutils import get_all_files_recursive
np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run script with mode and optional start_id.")
    parser.add_argument("--mode", type=str, required=True,
                         choices=["train", "eval", "test", "gridsearch"], help="Mode to run")
    parser.add_argument("--start_id", type=int, help="Start ID for processing (required for eval)")

    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    first_values = []
    if args.mode == "eval":
        cfg.MODE = 1
        if args.start_id is None:
            # take the top 10 occurences of first note
            print("Reading startprobabilities.csv")
            with open('./RESULTS/' + cfg.BACKUP + "/"
                       + cfg.BACKUP + '_startprobability.txt', 'r',
                       encoding = 'utf-8') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        number_str = parts[1].replace(':', '')  # Remove the colon
                        first_values.append(int(number_str))
        else:
            first_values.append(args.start_id)
        print(f"Evaluating from ID {args.start_id}...")
        # evaluation logic
    elif args.mode == "train":
        cfg.MODE = 0
        print("Training...")
    elif args.mode == "gridsearch":
        cfg.MODE = 4
        print("Grid Search")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os_type = platform.system()
    if os_type == 'Darwin':
        DEVICE_TYPE     =   "mps"
    else:
        DEVICE_TYPE     =   "cuda"

    accents_collect = np.zeros((24), dtype='int32')
    start_collect = np.zeros((cfg.VOCAB_SIZE), dtype = 'int32')
    ################################
    end_seq = False
    NUM_PATCH = ((cfg.MAX_SEQ_LENGTH - cfg.PATCH)//cfg.STRIDE) + 1
    device = torch.device(DEVICE_TYPE)
    decoder = DecoderAPE(device, cfg.D_MODEL, cfg.VOCAB_SIZE, cfg.FFN_HIDDEN, cfg.MAX_SEQ_LENGTH,
                         cfg.NUM_HEADS, cfg.DROP_PROB, cfg.NUM_LAYERS).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr = cfg.LEARNING_RATE)
    if cfg.SCHEDULER == 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               patience=5,
                                                               factor=0.5)
    embedding_layer = nn.Embedding(num_embeddings = cfg.VOCAB_SIZE,
                                        embedding_dim = cfg.D_MODEL).to(device)
    pos_enc = get_positional_encoding(cfg.MAX_SEQ_LENGTH, cfg.D_MODEL).to(device)
    casualmask = torch.full([NUM_PATCH, NUM_PATCH], float('-inf'))
    casualmask = torch.triu(casualmask, diagonal = 1).to(device)
    casual_mask_expanded = casualmask.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
    casual_mask_expanded = casual_mask_expanded.expand(cfg.BATCH, cfg.NUM_HEADS,
                                                       cfg.MAX_SEQ_LENGTH,
                                                       cfg.MAX_SEQ_LENGTH)

    if cfg.MODE in (0, 2, 3, 4):
        create_dir('./RESULTS/')
        create_dir('./RESULTS/' + cfg.BACKUP)
        create_dir('./RESULTS/' + cfg.BACKUP + "/gp5")
        shutil.copy("./config.py", "./RESULTS/" + cfg.BACKUP + "/" + cfg.BACKUP + ".py")
        training_src_encoder_1 = np.zeros((1 * cfg.MAX_SEQ_LENGTH), dtype = 'int32')
        training_note_encoder_1 = np.zeros((cfg.NUM_SEQUENCE * cfg.MAX_SEQ_LENGTH),
                                           dtype = 'int32')
        training_beat_encoder_1 = np.zeros((cfg.NUM_SEQUENCE * cfg.MAX_SEQ_LENGTH),
                                           dtype = 'int32')
        training_beattype_encoder_1 = np.zeros((cfg.NUM_SEQUENCE * cfg.MAX_SEQ_LENGTH),
                                               dtype = 'int32')
        GPROFOLDER = './gprofiles/'
        L = 0
        L_MAX = cfg.MAX_SEQ_LENGTH * cfg.NUM_SEQUENCE - 1
        nval = 0
        SEQ_LEN = cfg.MAX_SEQ_LENGTH       # e.g., 32
        OVERLAP = cfg.OVERLAP               # e.g., 8
        STEP = SEQ_LEN - OVERLAP           # how much to move forward each time
        training_src_encoder_1 = training_src_encoder_1.reshape(1, cfg.MAX_SEQ_LENGTH)
        NUM_S = 0
        FILE_IND = 0
        start_value = None
        end_value = None
        for f in cfg.TRAINING:
            filenamelist = get_all_files_recursive(GPROFOLDER + f)
            for filename in filenamelist:
                training_src = []
                file_path = filename
                # Check if it is a file (not a directory)
                if os.path.isfile(file_path) and not end_seq:
                    if cfg.MODE != 4:
                        print(f"File: {file_path}")
                    song = readgpro(str(file_path))
                    tuning = guitarinfo(song)

                    for track in song.tracks:
                        if not end_seq:
                            # Map values to Genaral MIDI.
                            print(f"Total measures : {len(track.measures)}")
                            start_measure = random.randint(0, len(track.measures))
                            for measure in track.measures[start_measure:]:
                                if not end_seq:
                                    for beat in measure.voices[0].beats:
                                        L = 0
                                        if not end_seq:
                                            for note_index, note in enumerate(beat.notes):
                                                training_src.append(tokenizer_1(note.value,
                                                                        note.string,
                                                                        note.beat.duration,
                                                                        note.effect.palmMute + 1))
                                                if L == 0:
                                                    if training_src[L] < cfg.BAR:
                                                        start_collect[training_src[L]] += 1
                                                training_note_encoder_1[nval] = note_prob(
                                                                                note.value,
                                                                                note.string)
                                                training_beat_encoder_1[nval] = beat_prob(
                                                                                note.beat.duration)
                                                training_beattype_encoder_1[nval] = \
                                                            beattype_prob(note.beat.duration.value)
                                                nval += 1
                                                L = min(L + 1, L_MAX)
                                                if note_index != 0:
                                                    training_src.append(cfg.BARRE_NOTE)
                                                    accents_collect[0] += 1
                                                    L = min(L + 1, L_MAX)

                                                if note.effect.isBend > 0:
                                                    if note.effect.bend.type == gp.BendType.bend:
                                                        training_src.append(cfg.BEND_NOTE_1)
                                                        accents_collect[1] += 1
                                                    elif note.effect.bend.type == \
                                                                gp.BendType.bendRelease:
                                                        training_src.append(cfg.BEND_NOTE_3)
                                                        accents_collect[3] += 1
                                                    elif note.effect.bend.type == \
                                                                gp.BendType.prebend:
                                                        points = note.effect.bend.points
                                                        if len(points) < 2:
                                                            training_src.append(cfg.BEND_NOTE_5)
                                                            accents_collect[5] += 1
                                                        start_val = points[0].value
                                                        next_val = points[1].value

                                                        # Check if first point starts bent
                                                        # and next one bends higher
                                                        if next_val > start_val > 0:
                                                            training_src.append(cfg.BEND_NOTE_6)
                                                            accents_collect[6] += 1
                                                        else:
                                                            training_src.append(cfg.BEND_NOTE_5)
                                                            accents_collect[5] += 1
                                                    elif note.effect.bend.type == \
                                                            gp.BendType.prebendRelease:
                                                        training_src.append(cfg.BEND_NOTE_7)
                                                        accents_collect[7] += 1
                                                    L = min(L + 1, L_MAX)

                                                if note.type.name == 'dead':
                                                    training_src.append(cfg.DEAD_NOTE)
                                                    accents_collect[19] += 1
                                                    L = min(L + 1, L_MAX)

                                                if beat.effect.isTremoloBar is True:
                                                    points = beat.effect.tremoloBar.points
                                                    if points:
                                                        start_value = points[0].value
                                                        end_value = points[-1].value

                                                    if start_value == 0 and end_value < 0:
                                                        training_src.append(cfg.TREM_BAR_1)
                                                        accents_collect[8] += 1

                                                    # Dip: starts neutral, dips, returns to neutral
                                                    elif start_value == 0 and end_value == 0 \
                                                        and any(p.value < 0 for p in points[1:-1]):
                                                        training_src.append(cfg.TREM_BAR_2)
                                                        accents_collect[9] += 1

                                                    # Pre-dive: starts low, returns to neutral
                                                    elif start_value < 0 and end_value != 0:
                                                        training_src.append(cfg.TREM_BAR_4)
                                                        accents_collect[11] += 1

                                                    elif start_value < 0 and end_value == 0:
                                                        training_src.append(cfg.TREM_BAR_5)
                                                        accents_collect[12] += 1
                                                    else:
                                                        training_src.append(cfg.TREM_BAR_3)
                                                        accents_collect[10] += 1

                                                    L = min(L + 1, L_MAX)

                                                if note.effect.slides:
                                                    if note.effect.slides[0].name ==\
                                                            'legatoSlideTo':
                                                        training_src.append(cfg.SLIDE_NOTE_1)
                                                        accents_collect[13] += 1
                                                    elif note.effect.slides[0].name ==\
                                                            'shiftSlideTo':
                                                        training_src.append(cfg.SLIDE_NOTE_2)
                                                        accents_collect[14] += 1
                                                    elif note.effect.slides[0].name ==\
                                                            'intoFromBelow':
                                                        training_src.append(cfg.SLIDE_NOTE_3)
                                                        accents_collect[15] += 1
                                                    elif note.effect.slides[0].name ==\
                                                            'intoFromAbove':
                                                        training_src.append(cfg.SLIDE_NOTE_4)
                                                        accents_collect[16] += 1
                                                    elif note.effect.slides[0].name ==\
                                                            'outDownwards':
                                                        training_src.append(cfg.SLIDE_NOTE_5)
                                                        accents_collect[17] += 1
                                                    elif note.effect.slides[0].name ==\
                                                          'outUpwards':
                                                        training_src.append(cfg.SLIDE_NOTE_6)
                                                        accents_collect[18] += 1
                                                    L = min(L + 1, L_MAX)

                                                if note.effect.hammer is True:
                                                    training_src.append(cfg.HAMMER)
                                                    accents_collect[20] += 1
                                                    L = min(L + 1, L_MAX)

                                                if note.effect.vibrato is True:
                                                    training_src.append(cfg.VIBRATO)
                                                    accents_collect[21] += 1
                                                    L = min(L + 1, L_MAX)

                                                if note.effect.isHarmonic is True:
                                                    training_src.append(cfg.HARMONIC_1)
                                                    accents_collect[22] += 1
                                                    L = min(L + 1, L_MAX)

                                                if note.effect.palmMute:
                                                    accents_collect[23] += 1

                                                if nval == cfg.NUM_SEQUENCE * \
                                                            cfg.MAX_SEQ_LENGTH - 1:
                                                    end_seq = True
                                                    print(f"Max number of sequences"
                                                          f"reached!{nval}")
                                                    break

                                    training_src.append(cfg.BAR)
                                    L = min(L + 1, L_MAX)

                if cfg.EOS_TRUE:
                    training_src.append(cfg.EOS)
                    remainder = len(training_src) % 32
                    if remainder != 0:
                        pad_length = 32 - remainder
                        training_src.extend([0] * pad_length)
                    long_tokens = training_src
                    num_chunks = (len(long_tokens) - OVERLAP) // STEP
                    overlapping_sequences = np.zeros((num_chunks, SEQ_LEN), dtype='int32')
                    # Fill in overlapping chunks
                    for i in range(num_chunks):
                        start = i * STEP
                        end = start + SEQ_LEN
                        overlapping_sequences[i] = long_tokens[start:end]
                training_src_encoder_1 = np.concatenate((training_src_encoder_1,
                                                          overlapping_sequences), axis=0)

        NUM_SEQUENCE = len(training_src_encoder_1)
        print(f"Sequence after overlap : {NUM_SEQUENCE}")
        training_tgt_decoder_1 = training_src_encoder_1.copy().astype(np.int64)
        training_tgt_notes = np.roll(training_tgt_decoder_1, shift=-1)
        training_tgt_notes = np.zeros_like(training_tgt_decoder_1)
        training_tgt_notes[:, :-1] = training_tgt_decoder_1[:, 1:]
        training_tgt_notes[:, -1] = 0  # pad token
        training_src_encoder_1 = torch.tensor(training_src_encoder_1)
        training_tgt_notes = torch.tensor(training_tgt_notes)
        # bool mask
        full_pad_mask = (training_src_encoder_1 == 0).unsqueeze(1).unsqueeze(2)
        full_pad_mask = full_pad_mask.expand(NUM_SEQUENCE, 1,
                                             cfg.MAX_SEQ_LENGTH, cfg.MAX_SEQ_LENGTH)
        # use -1e9 instead of -inf
        full_pad_mask = full_pad_mask.to(dtype=torch.float32) * float('-1e9')
        full_pad_mask = full_pad_mask.to(device)
        train_src, temp_src, train_tgt, temp_tgt = train_test_split(
            training_src_encoder_1,
            training_tgt_notes,
            test_size=0.2,  # 20% reserved for val+test
            random_state=42
        )
        val_src, test_src, val_tgt, test_tgt = train_test_split(
            temp_src,
            temp_tgt,
            test_size=0.5,  # split the 20% evenly → 10% val, 10% test
            random_state=42
        )
        train_src_tensor = train_src.detach().clone()
        train_tgt_tensor = train_tgt.detach().clone()
        val_src_tensor = val_src.detach().clone()
        val_tgt_tensor = val_tgt.detach().clone()
        test_src_tensor = test_src.detach().clone()
        test_tgt_tensor = test_tgt.detach().clone()
        del training_tgt_decoder_1

        sorted_indices = np.argsort(-start_collect)
        # print("Source")
        # print(f"{training_tgt_notes}")
        # plot notes
        training_note_encoder_1 = training_note_encoder_1[training_note_encoder_1 != 0]
        training_beat_encoder_1 = training_beat_encoder_1[training_beat_encoder_1 != 0]
        training_beattype_encoder_1 = training_beattype_encoder_1[training_beattype_encoder_1 != 0]
        bincounts = np.bincount(training_note_encoder_1, minlength=13)[1:]
        bincountsbeats = np.bincount(training_beat_encoder_1, minlength=15)[1:]
        bincountsbeattype = np.bincount(training_beattype_encoder_1, minlength=8)[1:]
        bincountsaccents = np.bincount(accents_collect, minlength=24)[1:]

        if cfg.MODE != 4:
            writebincount(bincounts, './RESULTS/'
                           + cfg.BACKUP + "/" + cfg.BACKUP + '_trainingprobability.txt')
            plotbar(cfg.labelsnotes, 'Occurance of Notes',
                     bincounts, './RESULTS/' + cfg.BACKUP + "/"
                       + cfg.BACKUP + '_trainingprobability.pdf')
            del training_note_encoder_1

            writebincount(bincountsbeats, './RESULTS/'
                           + cfg.BACKUP + "/" + cfg.BACKUP +
                             '_trainingbeatprobability.txt')
            plotbar(cfg.labelsbeats, 'Occurance of Beats',
                     bincountsbeats, './RESULTS/' + cfg.BACKUP + "/"
                       + cfg.BACKUP + '_trainingbeatprobability.pdf')
            del training_beat_encoder_1

            writebincount(bincountsbeattype, './RESULTS/'
                           + cfg.BACKUP + "/" + cfg.BACKUP + '_trainingbeattypeprobability.txt')
            plotbar(cfg.labelsbeattype, 'Occurance of Beat Type',
                     bincountsbeattype, './RESULTS/' + cfg.BACKUP + "/"
                       + cfg.BACKUP + '_trainingbeattypeprobability.pdf')
            del training_beattype_encoder_1

            writebincount(accents_collect, './RESULTS/'
                           + cfg.BACKUP + "/" + cfg.BACKUP + '_trainingaccentprobability.txt')
            plotbarlog(cfg.labelsaccents, 'Occurance of Accents',
                       accents_collect, './RESULTS/' + cfg.BACKUP + "/"
                         + cfg.BACKUP + '_trainingaccentprobability.pdf')
            del accents_collect

            top_indices = sorted_indices[:10]
            top_values = start_collect[top_indices]
            top_labels = [str(i) for i in top_indices]
            writebincount2(top_labels, start_collect[sorted_indices[:10]],
                            './RESULTS/' + cfg.BACKUP + "/" + cfg.BACKUP + '_startprobability.txt')
            plotbar(top_labels, 'Occurance of First note', top_values,
                     './RESULTS/' + cfg.BACKUP + "/" + cfg.BACKUP + '_startprobability.pdf')

        ITERATION = 0
        lossplot = []
        ce_lossplot = []
        rep_lossplot =[]
        val_lossplot = []
        seq_lossplot = []
        penalize_tokens = [cfg.BEND_NOTE_1, cfg.BEND_NOTE_2, cfg.BEND_NOTE_3,
                           cfg.BEND_NOTE_4,
                           cfg.BEND_NOTE_5, cfg.BEND_NOTE_6, cfg.BEND_NOTE_7,
                           cfg.TREM_BAR_1, cfg.TREM_BAR_2, cfg.TREM_BAR_3,
                           cfg.TREM_BAR_4, cfg.TREM_BAR_5,
                           cfg.DEAD_NOTE, cfg.SLIDE_NOTE_1, cfg.SLIDE_NOTE_2,
                           cfg.SLIDE_NOTE_3,
                           cfg.SLIDE_NOTE_4, cfg.SLIDE_NOTE_5, cfg.SLIDE_NOTE_6,
                           cfg.BOS, cfg.VIBRATO, cfg.HAMMER, cfg.HARMONIC_1]
        criterion = RepetitionPenaltyLossForSpecificTokens(
            label_smoothing=cfg.SMOOTHING,
            repetition_penalty_weight=cfg.LAMBDA,
            sequence_penalty_weight=cfg.DELTA,
            ngram_size=3,
            penalize_tokens=penalize_tokens
        )
        sequence_memory = SequenceMemory(NUM_SEQUENCE).to(device)

        # Wrap your full data into a TensorDataset
        dataset = TensorDataset(train_src_tensor, train_tgt_tensor)
        loader = DataLoader(dataset, batch_size=cfg.BATCH, shuffle=True, drop_last=True)
        sequence_ids = torch.arange(len(dataset))

        dataset_val = TensorDataset(val_src_tensor, val_tgt_tensor)
        loader_val = DataLoader(dataset_val, batch_size=cfg.BATCH, shuffle=True)

        dataset_test = TensorDataset(test_src_tensor, test_tgt_tensor)
        loader_test = DataLoader(dataset_test, batch_size=cfg.BATCH, shuffle=True)

        if cfg.MODE == 3:
            print("Loading saved file", cfg.SAVE)
            checkpoint = torch.load('./RESULTS/'+ cfg.BACKUP + "/"
                                     + cfg.BACKUP +'.pth', map_location=torch.device(DEVICE_TYPE))
            decoder.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            embedding_layer.load_state_dict(checkpoint['embedding_state_dict'])
            ITERATION = checkpoint['epoch']
        print("\nTotal epochs : ", cfg.EPOCHS)

        while ITERATION < cfg.EPOCHS:
            decoder.train()
            epoch_loss = 0.0  # track epoch loss
            epoch_rep_loss = 0.0
            epoch_seq_loss = 0.0
            epoch_ce_loss = 0.0
            batch_count = 0
            kl_loss = 0.0
            total_tokens = 0
            prev_sequence_embedding = None
            for batch_idx, (batch_token_ids, batch_target) in enumerate(loader):
                #if len(batch_token_ids) != cfg.BATCH:
                #    break
                # print(batch_count, end = " ")
                batch_token_ids = batch_token_ids.to(device)
                batch_target = batch_target.to(device)
                start = batch_idx * cfg.BATCH
                end = start + batch_token_ids.size(0)  # actual batch size
                # Prepare `inputs` tensor (e.g., for teacher forcing or decoder input)
                pad_mask = full_pad_mask[start:end]
                mask = casual_mask_expanded + pad_mask

                optimizer.zero_grad()
                # Embeddings
                embeddings = embedding_layer(batch_token_ids)
                input_embeddings = embeddings + pos_enc[:batch_token_ids.size(1)]
                # Forward
                logits = decoder(input_embeddings, mask)
                # Flatten target
                batch_target = batch_target.view(-1)
                batch_seq_ids = sequence_ids[start:end].to(device)
                seq_mem = sequence_memory(batch_seq_ids)
                # Compute loss
                loss, ce_loss, rep_loss, kl_loss, prev_sequence_embedding = criterion(logits,
                                                                            batch_target,
                                                                            prev_sequence_embedding,
                                                                            seq_mem)

                loss.backward()
                optimizer.step()

                # Log and accumulate loss
                num_tokens = (batch_target != 0).sum().item()
                epoch_loss += loss.item()
                epoch_ce_loss += ce_loss.item() * num_tokens
                epoch_rep_loss += rep_loss.item()
                epoch_seq_loss += kl_loss.item()
                batch_count += 1
                total_tokens += num_tokens
                # print("batch count : ", batch_count)

            avg_loss = epoch_loss / batch_count
            avg_ce_loss = epoch_ce_loss / total_tokens
            avg_rep_loss = epoch_rep_loss / batch_count
            avg_seq_loss = epoch_seq_loss / batch_count
            val_loss = validation(loader_val, device, optimizer, embedding_layer, decoder, pos_enc)
            test_loss = None
            if cfg.SCHEDULER == 1:
                scheduler.step(val_loss)
            if ITERATION % 5 == 0:
                test_loss = testing(loader_test, device, embedding_layer, decoder, pos_enc)
            print(f"{ITERATION + 1} : main :{round(avg_loss, 4)}, ce :{round(avg_ce_loss, 4)}, "
                  f"rep :{round(avg_rep_loss, 4)}, seq :{round(avg_seq_loss, 4)}, "
                  f"validation :{round(val_loss, 4)}, "
                  f"testing :{round(test_loss, 4)}")
            ITERATION += 1
            lossplot.append(avg_loss)
            ce_lossplot.append(avg_ce_loss)
            rep_lossplot.append(avg_rep_loss)
            val_lossplot.append(val_loss)
            seq_lossplot.append(avg_seq_loss)

            if avg_loss < cfg.CONVERGENCE:
                print("Convergence criteria reached!")
                break

            if ITERATION + 1 % cfg.SAVE_EVERY == 0 and cfg.MODE != 4:
                checkpoint = {
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'embedding_state_dict': embedding_layer.state_dict(),
                'epoch': ITERATION,  # Optionally save the epoch number
                'loss': loss     # Optionally save the loss value
                }
                torch.save(checkpoint, './RESULTS/'+ cfg.BACKUP + "/" + cfg.BACKUP +'.pth')
                print("Checkpoint saved")

        with open('./RESULTS/'+ cfg.BACKUP + "/" + cfg.BACKUP + '.csv', mode='a',
                  encoding='utf-8',newline='') as lossfile:
            writer = csv.writer(lossfile)
            writer.writerows(zip(lossplot, ce_lossplot, rep_lossplot, val_lossplot, seq_lossplot))
        checkpoint = {
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'embedding_state_dict': embedding_layer.state_dict(),
        'epoch': ITERATION,  # Optionally save the epoch number
        'loss': loss     # Optionally save the loss value
        }
        torch.save(checkpoint, './RESULTS/'+ cfg.BACKUP + "/" + cfg.BACKUP +'.pth')
        if cfg.MODE != 4:
            plot_multiple([lossplot, ce_lossplot, rep_lossplot,
                            val_lossplot, seq_lossplot], ["Total loss",
                                                        "Training loss",
                                                        "Repetition loss",
                                                        "Validation_loss",
                                                        "Sequence loss"],loss.item(),
                                                        './RESULTS/' + cfg.BACKUP +
                                                        "/" + cfg.BACKUP + '_loss.pdf')

    if cfg.MODE in (1, 2):
        create_dir('./RESULTS/' + cfg.BACKUP + "/gp5")
        checkpoint = torch.load('./RESULTS/'+ cfg.BACKUP + "/"
                                 + cfg.BACKUP +'.pth', map_location=torch.device(DEVICE_TYPE))
        decoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        embedding_layer.load_state_dict(checkpoint['embedding_state_dict'])
        print(f"Loss : ', {checkpoint['loss'].item()}")
        print(f"Epochs : ', {checkpoint['epoch']}")
        decoder.eval()
        embedding_layer.eval()

        song_notes_collect = []
        song_beats_collect = []
        song_accent_collect = []
        song_beattype_collect = []
        for sid, startid in enumerate(first_values):
            if sid >= cfg.TEST_NUM:
                break
            print("\nTesting with start-id :",startid)
            dummy_in = inference(startid, device, decoder, embedding_layer, pos_enc, casualmask)

            m = 0
            song = gp.models.Song()
            song.title = cfg.SAVE
            song.artist = "DjentleViBe"
            song.tempo = 120  # Set the tempo
            song.tracks[0].name = "Guitar"
            song.tracks[0].channel.instrument = 30
            song_collect = []
            song_notes = []
            song_beats = []
            song_accent = []
            song_beattype = []

            while m < cfg.TEST_TRIES:
                noteval = []
                notetypeval = []
                stringnum = []
                beatval = []
                palmval = []
                accentval = []

                for ind, dummy in enumerate(dummy_in[m]):
                    if cfg.VERBOSE == 1:
                        print(f"{ind + 1:02}", end=' ')
                    note, notetype, string, beat, palm, beatnum, accent = detokenizer_1(dummy)
                    noteval.append(note)
                    notetypeval.append(notetype)
                    stringnum.append(string)
                    beatval.append(beat)
                    palmval.append(palm)
                    song_accent.append(accent)
                    song_notes.append(note_prob(note, string))
                    song_beats.append(beatnum)
                    song_beattype.append(beattype_prob(beat["duration"]))
                song_collect.append(makegpro(cfg.SAVE, noteval, stringnum, beatval, palmval))
                song.tracks[0].measures.append(song_collect[m].tracks[0].measures[0])
                song.tracks[0].strings[0].value = cfg.TUNING[0]
                song.tracks[0].strings[1].value = cfg.TUNING[1]
                song.tracks[0].strings[2].value = cfg.TUNING[2]
                song.tracks[0].strings[3].value = cfg.TUNING[3]
                song.tracks[0].strings[4].value = cfg.TUNING[4]
                song.tracks[0].strings[5].value = cfg.TUNING[5]
                if m == 0:
                    del song.tracks[0].measures[0]
                m += 1
                writegpro(cfg.SAVE + "_" + str(startid), song)
                song_notes_collect.append(song_notes)
                song_beats_collect.append(song_beats)
                song_accent_collect.append(song_accent)
                song_beattype_collect.append(song_beattype)

        flattened_notes = [note for song_notes in song_notes_collect
                            for note in song_notes if note != 100]
        flattened_beats = [beat for song_beats in song_beats_collect
                            for beat in song_beats]
        flattened_accents = [accent - 1 for song_accent in song_accent_collect
                              for accent in song_accent if accent != 0]
        flattened_beattype = [bt for song_beattype in song_beattype_collect
                               for bt in song_beattype]

        bincounts_inf = np.bincount(flattened_notes, minlength=13)[1:]
        bincountsbeats_inf = np.bincount(flattened_beats, minlength=15)[1:]
        bincountsaccent_inf = np.bincount(flattened_accents, minlength=25)[1:]
        bincountsbeattype_inf = np.bincount(flattened_beattype, minlength=8)[1:]

        writebincount(bincounts_inf, './RESULTS/' + cfg.BACKUP +
                      "/" + cfg.SAVE + '_inferenceprobability.pdf')
        bincounts_train = readbincount('./RESULTS/' + cfg.BACKUP +
                                       "/" + cfg.BACKUP + '_trainingprobability.txt')
        bincountsbeats_train = readbincount('./RESULTS/' + cfg.BACKUP +
                                            "/" + cfg.BACKUP + 
                                            '_trainingbeatprobability.txt')
        bincountsbeattype_train = readbincount('./RESULTS/' + cfg.BACKUP +
                                               "/" + cfg.BACKUP +
                                               '_trainingbeattypeprobability.txt')
        bincountsaccents_train = readbincount('./RESULTS/' + cfg.BACKUP +
                                              "/" + cfg.BACKUP +
                                              '_trainingaccentprobability.txt')

        if cfg.MODE != 4:
            plotbar(cfg.labelsnotes, 'Occurance of Notes', bincounts_inf,
                    './RESULTS/' + cfg.BACKUP + "/" + cfg.SAVE +
                    '_inferenceprobabilitynotes.pdf')
            plotbar(cfg.labelsbeats, 'Occurance of Beats', bincountsbeats_inf,
                    './RESULTS/' + cfg.BACKUP + "/" + cfg.SAVE +
                    '_inferenceprobabilitybeats.pdf')
            plotbar(cfg.labelsaccents, 'Occurance of Accents', bincountsaccent_inf,
                    './RESULTS/' + cfg.BACKUP + "/" + cfg.SAVE +
                    '_inferenceprobabilityaccent.pdf')
            plotbar(cfg.labelsbeattype, 'Occurance of Beat Type', bincountsbeattype_inf,
                    './RESULTS/' + cfg.BACKUP + "/" + cfg.SAVE +
                    '_inferenceprobabilitybeattype.pdf')

            KLD = kldivergence(bincounts_train, bincounts_inf)
            plotbar_dual('Notes', cfg.labelsnotes, bincounts_train,
                         bincounts_inf, KLD, './RESULTS/' + cfg.BACKUP +
                         "/" + cfg.SAVE + '_compareprobabilitynotes.pdf')

            KLD = kldivergence(bincountsbeats_train, bincountsbeats_inf)
            plotbar_dual('Beats', cfg.labelsbeats, bincountsbeats_train,
                         bincountsbeats_inf, KLD, './RESULTS/' + cfg.BACKUP +
                         "/" + cfg.SAVE + '_compareprobabilitybeats.pdf')

            KLD = kldivergence(bincountsaccents_train, bincountsaccent_inf)
            plotbar_dual('Accents', cfg.labelsaccents, bincountsaccents_train,
                         bincountsaccent_inf, KLD, './RESULTS/' + cfg.BACKUP +
                         "/" + cfg.SAVE + '_compareprobabilityaccent.pdf')

            KLD = kldivergence(bincountsbeattype_train, bincountsbeattype_inf)
            plotbar_dual('Beat type', cfg.labelsbeattype, bincountsbeattype_train,
                          bincountsbeattype_inf, KLD, './RESULTS/' + cfg.BACKUP +
                          "/" + cfg.SAVE + '_compareprobabilitybeattype.pdf')

        #combinepng('./RESULTS/' + cfg.BACKUP + "/" + cfg.SAVE + '_compareprobabilitynotes.pdf',
        #           './RESULTS/' + cfg.BACKUP + "/" + cfg.SAVE + '_compareprobabilitybeats.pdf',
        #          './RESULTS/' + cfg.BACKUP + "/" + cfg.SAVE + '_compareprobability.pdf')

    print("Finished")
