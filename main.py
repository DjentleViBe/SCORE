"""Training riffs using ML"""
import os
import sys
import shutil
import csv
import numpy as np
import torch
from torch import nn
from preprocess import readgpro, guitarinfo, get_positional_encoding, create_dir
from postprocess import plot, plotbar, plotbar_dual, decoder_inference, makegpro, writegpro, writebincount, readbincount, KLDivergence
from encoding import tokenizer_1, note_prob
from decoding import detokenizer_1
from _decoder.decoder import DecoderAPE
import config as cfg
from testing import inference
import guitarpro as gp
np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':

    if os.name == 'posix':
        DEVICE_TYPE     =   "mps"
    else:
        DEVICE_TYPE     =   "cuda"
    ################################
    NUM_PATCH = ((cfg.MAX_SEQ_LENGTH - cfg.PATCH)//cfg.STRIDE) + 1
    device = torch.device(DEVICE_TYPE)
    decoder = DecoderAPE(device, cfg.D_MODEL, cfg.VOCAB_SIZE, cfg.FFN_HIDDEN, cfg.MAX_SEQ_LENGTH,
                         cfg.NUM_HEADS, cfg.DROP_PROB, cfg.NUM_LAYERS).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr = cfg.LEARNING_RATE)
    if cfg.SCHEDULER == 1:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = cfg.SCHEDULER_SIZE, gamma=0.1)
    embedding_layer = nn.Embedding(num_embeddings = cfg.VOCAB_SIZE,
                                        embedding_dim = cfg.D_MODEL).to(device)
    pos_enc = get_positional_encoding(cfg.MAX_SEQ_LENGTH, cfg.D_MODEL).to(device)
    mask = torch.full([NUM_PATCH, NUM_PATCH], float('-inf'))
    mask = torch.triu(mask, diagonal = 1).to(device)

    if cfg.MODE in (0, 2, 3):
        create_dir('./RESULTS/')
        create_dir('./RESULTS/' + cfg.BACKUP)
        shutil.copy("./config.py", "./RESULTS/" + cfg.BACKUP + "/" + cfg.BACKUP + ".py")
        training_src_encoder_1 = np.zeros((cfg.BATCH * cfg.MAX_SEQ_LENGTH), dtype = 'int32')
        training_note_encoder_1 = np.zeros((cfg.BATCH * cfg.MAX_SEQ_LENGTH), dtype = 'int32')
        GPROFOLDER = './gprofiles/'
        L = 0
        N = 0
        for f in cfg.TRAINING:
            for filename in os.listdir(GPROFOLDER + f):
                file_path = os.path.join(GPROFOLDER + f, filename)

                # Check if it is a file (not a directory)
                if os.path.isfile(file_path):
                    print(f"File: {file_path}")
                    song = readgpro(str(file_path))
                    tuning = guitarinfo(song)

                    for track in song.tracks:
                        # Map values to Genaral MIDI.
                        for measure in track.measures:
                            training_src_encoder_1[L] = cfg.BOS
                            L += 1
                            for beat in measure.voices[0].beats:
                                for note_index, note in enumerate(beat.notes):
                                    training_src_encoder_1[L] = tokenizer_1(note.value,
                                                                        note.string,
                                                                        note.beat.duration,
                                                                        note.effect.palmMute + 1)
                                    training_note_encoder_1[N] = note_prob(note.value, note.string)
                                    N += 1
                                    L += 1
                                    if note_index != 0:
                                        training_src_encoder_1[L] = cfg.BARRE_NOTE
                                        L += 1

                                    if note.effect.isBend > 0:
                                        if note.effect.bend.type.value == 1:
                                            training_src_encoder_1[L] = cfg.BEND_NOTE_1
                                        elif note.effect.bend.type.value == 2:
                                            training_src_encoder_1[L] = cfg.BEND_NOTE_2
                                        elif note.effect.bend.type.value == 3:
                                            training_src_encoder_1[L] = cfg.BEND_NOTE_3
                                        elif note.effect.bend.type.value == 4:
                                            training_src_encoder_1[L] = cfg.BEND_NOTE_4
                                        elif note.effect.bend.type.value == 5:
                                            training_src_encoder_1[L] = cfg.BEND_NOTE_5
                                        elif note.effect.bend.type.value == 6:
                                            training_src_encoder_1[L] = cfg.BEND_NOTE_6
                                        elif note.effect.bend.type.value == 7:
                                            training_src_encoder_1[L] = cfg.BEND_NOTE_7
                                        L += 1
                                    
                                    if note.type.name == 'dead':
                                        training_src_encoder_1[L] = cfg.DEAD_NOTE
                                        L += 1

                                    if beat.effect.isTremoloBar is True:
                                        if beat.effect.tremoloBar.type.value == 1:
                                            training_src_encoder_1[L] = cfg.TREM_BAR_1
                                        elif beat.effect.tremoloBar.type.value == 2:
                                            training_src_encoder_1[L] = cfg.TREM_BAR_2
                                        elif beat.effect.tremoloBar.type.value == 3:
                                            training_src_encoder_1[L] = cfg.TREM_BAR_3
                                        elif beat.effect.tremoloBar.type.value == 4:
                                            training_src_encoder_1[L] = cfg.TREM_BAR_4
                                        elif beat.effect.tremoloBar.type.value == 5:
                                            training_src_encoder_1[L] = cfg.TREM_BAR_5
                                        L += 1

                                    if note.effect.slides:
                                        if note.effect.slides[0].name == 'legatoSlideTo':
                                            training_src_encoder_1[L] = cfg.SLIDE_NOTE_1
                                        elif note.effect.slides[0].name == 'shiftSlideTo':
                                            training_src_encoder_1[L] = cfg.SLIDE_NOTE_2
                                        elif note.effect.slides[0].name == 'intoFromBelow':
                                            training_src_encoder_1[L] = cfg.SLIDE_NOTE_3
                                        elif note.effect.slides[0].name == 'intoFromAbove':
                                            training_src_encoder_1[L] = cfg.SLIDE_NOTE_4
                                        elif note.effect.slides[0].name == 'outDownwards':
                                            training_src_encoder_1[L] = cfg.SLIDE_NOTE_5
                                        elif note.effect.slides[0].name == 'outUpwards':
                                            training_src_encoder_1[L] = cfg.SLIDE_NOTE_6
                                        L += 1
                                    
                                    if note.effect.hammer == True:
                                        training_src_encoder_1[L] = cfg.HAMMER
                                        L += 1

                                    if note.effect.vibrato == True:
                                        training_src_encoder_1[L] = cfg.VIBRATO
                                        L += 1

                                    if note.effect.isHarmonic == True:
                                        training_src_encoder_1[L] = cfg.HARMONIC_1
                                        L += 1
                                                        
                            if cfg.EOS_TRUE:
                                training_src_encoder_1[L] = cfg.EOS
                                L += 1
        training_tgt_decoder_1 = training_src_encoder_1.copy().astype(np.int64)
        training_src_encoder_1 = training_src_encoder_1.reshape(cfg.BATCH, cfg.MAX_SEQ_LENGTH)
        training_tgt_notes = np.roll(training_tgt_decoder_1, shift=-1)
        training_tgt_notes = training_tgt_notes.reshape(cfg.BATCH, cfg.MAX_SEQ_LENGTH)
        del training_tgt_decoder_1
        print("Source")
        print(f"{training_tgt_notes}")
        # plot notes
        training_note_encoder_1 = training_note_encoder_1[training_note_encoder_1 != 0]
        bincounts = np.bincount(training_note_encoder_1, minlength=12)[1:]
        writebincount(bincounts, './RESULTS/' + cfg.BACKUP + "/" + cfg.BACKUP + '_trainingprobability.txt')
        plotbar(bincounts, './RESULTS/' + cfg.BACKUP + "/" + cfg.BACKUP + '_trainingprobability.png')
        del training_note_encoder_1
        ITERATION = 0
        criterion = torch.nn.MSELoss()
        lossplot = []
        loss_fn = nn.CrossEntropyLoss()

        token_ids = torch.tensor(training_src_encoder_1).to(device)
        target = torch.from_numpy(training_tgt_notes).to(device)

        if cfg.MODE == 3:
            print("Loading saved file", cfg.SAVE)
            checkpoint = torch.load('./RESULTS/'+ cfg.BACKUP + "/" + cfg.BACKUP +'.pth', map_location=torch.device(DEVICE_TYPE))
            decoder.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            embedding_layer.load_state_dict(checkpoint['embedding_state_dict'])
            ITERATION = checkpoint['epoch']
        print(cfg.EPOCHS)
        while ITERATION <= cfg.EPOCHS:
            decoder.train()

            optimizer.zero_grad()
            embeddings = embedding_layer(token_ids)
            input_embeddings = embeddings + pos_enc

            logits = decoder(input_embeddings, mask)
            # Flatten logits to (batch_size * seq_length, d_vocab)
            logits = logits.view(-1, cfg.VOCAB_SIZE)
            
            # scaled_logits = logits / cfg.TEMPERATURE

            target = target.view(-1)

            loss = loss_fn(logits, target)

            print(f"{ITERATION + 1} : {loss.item()}")
            loss.backward()
            optimizer.step()
            if cfg.SCHEDULER == 1:
                scheduler.step()

            ITERATION += 1
            lossplot.append(loss.item())

            if loss.item() < cfg.CONVERGENCE:
                print("Convergence criteria reached!")
                break

            if ITERATION % cfg.SAVE_EVERY == 0:
                checkpoint = {
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'embedding_state_dict': embedding_layer.state_dict(),
                'epoch': ITERATION,  # Optionally save the epoch number
                'loss': loss     # Optionally save the loss value
                }
                torch.save(checkpoint, './RESULTS/'+ cfg.BACKUP + "/" + cfg.BACKUP +'.pth')
                print("Checkpoint saved")
        
        with open('./RESULTS/'+ cfg.BACKUP + "/" + cfg.BACKUP + '.csv', mode='a', newline='') as lossfile:
            writer = csv.writer(lossfile)
            writer.writerows([[value] for value in lossplot])
        checkpoint = {
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'embedding_state_dict': embedding_layer.state_dict(),
        'epoch': ITERATION,  # Optionally save the epoch number
        'loss': loss     # Optionally save the loss value
        }
        torch.save(checkpoint, './RESULTS/'+ cfg.BACKUP + "/" + cfg.BACKUP +'.pth')
        plot(lossplot, './RESULTS/' + cfg.BACKUP + "/" + cfg.BACKUP + '.png')

    if cfg.MODE in (1, 2):
        checkpoint = torch.load('./RESULTS/'+ cfg.BACKUP + "/" + cfg.BACKUP +'.pth', map_location=torch.device(DEVICE_TYPE))
        decoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        embedding_layer.load_state_dict(checkpoint['embedding_state_dict'])
        print(f"Loss : ', {checkpoint['loss'].item()}")
        print(f"Epochs : ', {checkpoint['epoch']}")
        decoder.eval()

        dummy_in = inference(device, decoder, embedding_layer, pos_enc, mask)

        m = 0
        song = gp.models.Song()
        song.title = cfg.SAVE
        song.artist = "DjentleViBe"
        song.tempo = 120  # Set the tempo
        song.tracks[0].name = "Guitar"
        song.tracks[0].channel.instrument = 30
        song.tracks[0].strings[0].value = 58
        song.tracks[0].strings[1].value = 53
        song.tracks[0].strings[2].value = 49
        song.tracks[0].strings[3].value = 44
        song.tracks[0].strings[4].value = 39
        song.tracks[0].strings[5].value = 32
        song_collect = []
        song_notes = []
        while m < cfg.TEST_TRIES:
            noteval = []
            notetypeval = []
            stringnum = []
            beatval = []
            palmval = []

            for ind, dummy in enumerate(dummy_in[m]):
                print(f"{ind + 1:02}", end=' ')
                note, notetype, string, beat, palm = detokenizer_1(dummy)
                noteval.append(note)
                notetypeval.append(notetype)
                stringnum.append(string)
                beatval.append(beat)
                palmval.append(palm)
                song_notes.append(note_prob(note, string))
            song_collect.append(makegpro(cfg.SAVE, noteval, stringnum, beatval, palmval))
            song.tracks[0].measures.append(song_collect[m].tracks[0].measures[0])
            m += 1
            print("")
            writegpro(cfg.SAVE, song)
        song_notes = [value for value in song_notes if value != 100]
        bincounts_inf = np.bincount(song_notes, minlength=12)[1:]
        writebincount(bincounts_inf, './RESULTS/' + cfg.BACKUP + "/" + cfg.BACKUP + '_inferenceprobability.png')
        bincounts_train = readbincount('./RESULTS/' + cfg.BACKUP + "/" + cfg.BACKUP + '_trainingprobability.txt')
        plotbar(bincounts_inf, './RESULTS/' + cfg.BACKUP + "/" + cfg.BACKUP + '_inferenceprobability.png')
        
    KLD = KLDivergence(bincounts_train, bincounts_inf)
    plotbar_dual(bincounts_train, bincounts_inf, KLD, './RESULTS/' + cfg.BACKUP + "/" + cfg.BACKUP + '_compareprobability.png')
    
    print("Finished")
