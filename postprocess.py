"""Post processing"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import guitarpro as gp
import math
import numpy as np
from matplotlib.patches import Patch
from config import (BACKUP, MAX_SEQ_LENGTH, EOS, BOS, BARRE_NOTE, MEASURE, BEND_NOTE_1, BEND_NOTE_2, BEND_NOTE_3,
BEND_NOTE_4, BEND_NOTE_5, BEND_NOTE_6, BEND_NOTE_7, TREM_BAR_1, TREM_BAR_2, TREM_BAR_3,
TREM_BAR_4, TREM_BAR_5, DEAD_NOTE, SLIDE_NOTE_1, SLIDE_NOTE_2, SLIDE_NOTE_3, SLIDE_NOTE_4, SLIDE_NOTE_5, SLIDE_NOTE_6,
HAMMER, VIBRATO, HARMONIC_1, 
TEMPERATURE, TEST_CRITERIA, PREDICTION_CRITERIA)
from inference import multinomial_sample

DEMAPPING_BEAT_DETYPE = {
    'Base---------------' : 1,
    'Triplet------------' : 2,
    'Quintuplet---------' : 3,
    'Sextuplet----------' : 4,
    'Septuplet----------' : 5,
    '9_Tuplets----------' : 6,
    '11_Tuplets---------' : 7,
    'Dotted - Base------' : 8,
    'Dotted - Triplet---' : 9,
    'Dotted - Quintuplet' : 10,
    'Dotted - Sextuplet-' : 11,
    'Dotted - Septuplet-' : 12,
    'Dotted - 9_Tuplets-' : 13,
    'Dotted - 11_Tuplets' : 14,
}

def plot_multiple(plot_collect, labels, text, filename):
    "Plots data and saves figure"
    plt.suptitle('SupervisedRiffGen')
    plt.title('Loss = ' + str(round(text, 4)))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    for i in range(0, len(plot_collect)):
        plt.plot(plot_collect[i], label=labels[i])
    plt.legend()
    plt.savefig(filename, dpi = 200)

    return 0
def plot(lossplot, text, filename):
    "Plots data and saves figure"
    plt.plot(lossplot)
    plt.suptitle('SupervisedRiffGen')
    plt.title('Loss = ' + str(round(text, 4)))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig(filename, dpi = 200)

    return 0

def plotbar(labels, title, counts, filename):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color='skyblue', edgecolor='black')
    # plt.xlabel('Notes', fontsize=12)
    plt.ylabel('Occurrences', fontsize=12)
    plt.title(title, fontsize=14)
    #plt.xticks(range(1, 13))
    plt.xticks(rotation=45) 

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi = 200)

    plt.close()

    return 0

def plotbar_dual(labels, counts1, counts2, KLD, filename, label1='Training', label2='Inference'):
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))  # label positions
    width = 0.4  # width of the bars

    # Primary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars1 = ax1.bar(x - width/2, counts1, width, label=label1, color='skyblue', edgecolor='black')
    # ax1.set_xlabel('Notes', fontsize=12)
    ax1.set_ylabel(f'Training', fontsize=12)
    ax1.tick_params(axis='y')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    fig.suptitle('Occurrences', fontsize=18)      # Title for the Axes
    ax1.set_title('KL_Divergence = ' + str(round(KLD, 4)), fontsize = 14)      # Subtitle for the entire figure
    plt.xticks(rotation=45)
    # Annotate primary bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=9)

    # Secondary axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, counts2, width, label=label2, color='salmon', edgecolor='black')
    ax2.set_ylabel(f'Inference', fontsize=12)
    ax2.tick_params(axis='y')

    # Annotate secondary bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=9)

    # Combine legends
    labels = [label1, label2]
    legend_patches = [
        Patch(facecolor='skyblue', edgecolor='black', label=label1),
        Patch(facecolor='salmon', edgecolor='black', label=label2)
    ]
    ax1.legend(handles=legend_patches, loc='upper right')

    # Grid on primary axis
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

    return 0

def decoder_search(decoder, dummy_in, embedding_layer, pos_enc, mask):
    embeddings = embedding_layer(dummy_in)
    output_eval = decoder(embeddings + pos_enc, mask)

    next_token_logits = output_eval[:, -1, :]
    scaled_logits = next_token_logits / TEMPERATURE
    
    probabilities = F.softmax(scaled_logits, dim=-1)
    

    # Select the next token (using greedy search here)
    if PREDICTION_CRITERIA == 1:
        next_token = torch.argmax(probabilities, dim=-1).unsqueeze(0)
    elif PREDICTION_CRITERIA == 2:
        k = 3  # Specify the beam width
        # Get the top k probabilities and their corresponding indices
        _, next_token = torch.topk(probabilities, k, dim=-1)
        
    elif PREDICTION_CRITERIA == 3:
        next_token = torch.multinomial(probabilities, num_samples=1).unsqueeze(0)
    elif PREDICTION_CRITERIA == 4:
        next_token = multinomial_sample(probabilities, num_samples=40)
    
    return next_token

def decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask, seq_lim):
    """Transformer Decoder"""
    with torch.no_grad():
        if PREDICTION_CRITERIA == 2:
            seq_lim = dummy_in.size(1)  # Sequence length limit
            beam_sequences = [dummy_in.clone()]  # Initialize beam with dummy input
            beam_scores = [0.0]  # Initialize scores with 0
            beam_width = 3
            dummy_in_collect = []
            for e_val in range (2, seq_lim):
                all_candidates = []
                # For each beam, perform a step in the beam search
                for i, beam in enumerate(beam_sequences):
                    # Perform decoding step (greedy or other search method)
                    next_token = decoder_search(decoder, beam, embedding_layer, pos_enc, mask)

                    # Extend each beam with the top k candidates
                    for bw in range(beam_width):
                        new_beam = beam.clone()  # Copy the current beam
                        new_beam[0][e_val] = next_token[0][bw]  # Update the token for this position
                        new_score = beam_scores[i] + next_token[0][bw].log()  # Update the score (log probability)
                        all_candidates.append((new_beam, new_score))

                # Sort all candidates by score and select the top k
                ordered_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
                beam_sequences = [candidate[0] for candidate in ordered_candidates[:beam_width]]
                beam_scores = [candidate[1] for candidate in ordered_candidates[:beam_width]]

                # Collect beams for the current step
                dummy_in_collect = [beam_sequences[bw] for bw in range(beam_width)]
            best_beam_index = torch.argmax(torch.tensor(beam_scores))
            dummy_in = beam_sequences[best_beam_index]
        else:
            if TEST_CRITERIA != 4:
                for e_val in range (2, seq_lim):
                    next_token = decoder_search(decoder, dummy_in, embedding_layer, pos_enc, mask)
                    # generated_sequence = dummy_in
                    dummy_in[0][e_val] = next_token
            else:
                e_val = 2
                trial = 0
                temperature_var = TEMPERATURE
                while e_val < seq_lim:
                    next_token = decoder_search(decoder, dummy_in, embedding_layer, pos_enc, mask)

                    if e_val != 2:
                        if next_token > BOS and dummy_in[0][e_val - 1] < BOS:
                            dummy_in[0][e_val] = next_token
                            e_val +=1
                        elif next_token < BOS:
                            dummy_in[0][e_val] = next_token
                            e_val +=1
                        else:
                            temperature_var /= 2
                            print("Trial -->", e_val, trial, next_token.cpu()[0][0], dummy_in.cpu()[0][e_val - 1])
                            trial += 1
                    else:
                        # generated_sequence = dummy_in
                        dummy_in[0][e_val] = next_token
                        e_val += 1

        print(dummy_in)
    return dummy_in

def getnotetype(notetype):
    """Return note type"""
    if 0 <= notetype <= 14:
        return 32, notetype
    elif 15 <= notetype <= 28:
        return 16, notetype - 14
    elif 29 <= notetype <= 40:
        return 8, notetype - 28
    elif 41 <= notetype <= 54:
        return 4, notetype - 40
    elif 55 <= notetype <= 68:
        return 2, notetype - 54
    elif 69 <= notetype <= 82:
        return 1, notetype - 68

def adjustmeasure(beat_collect):
    """calculate the total measure length"""
    measure_length = 0
    for b, beat_val in enumerate(beat_collect[1:]):
        measure_length += 4 / getnotetype(beat_val)[0]
    return math.ceil(measure_length)

def makegpro(titlename, noteval, stringnum, beatval, palmval):
    """Generate gpro file"""
    # read bend and tremolo templates
    song_trem_1 = gp.parse('./gprofiles/trem_1.gp5')
    song_trem_2 = gp.parse('./gprofiles/trem_2.gp5')
    song_trem_4 = gp.parse('./gprofiles/trem_4.gp5')
    song_trem_5 = gp.parse('./gprofiles/trem_5.gp5')
    trem1_beat = song_trem_1.tracks[0].measures[0].voices[0].beats[0]
    trem2_beat = song_trem_2.tracks[0].measures[0].voices[0].beats[0]
    trem4_beat = song_trem_4.tracks[0].measures[0].voices[0].beats[0]
    trem5_beat = song_trem_5.tracks[0].measures[0].voices[0].beats[0]

    song_bend_1 = gp.parse('./gprofiles/bend_1.gp5')
    song_bend_2 = gp.parse('./gprofiles/bend_2.gp5')
    song_bend_3 = gp.parse('./gprofiles/bend_3.gp5')
    song_bend_4 = gp.parse('./gprofiles/bend_4.gp5')
    song_bend_5 = gp.parse('./gprofiles/bend_5.gp5')
    song_bend_6 = gp.parse('./gprofiles/bend_6.gp5')
    song_bend_7 = gp.parse('./gprofiles/bend_7.gp5')
    bend1_beat = song_bend_1.tracks[0].measures[0].voices[0].beats[0]
    bend2_beat = song_bend_2.tracks[0].measures[0].voices[0].beats[1]
    bend3_beat = song_bend_3.tracks[0].measures[0].voices[0].beats[0]
    bend4_beat = song_bend_4.tracks[0].measures[0].voices[0].beats[1]
    bend5_beat = song_bend_5.tracks[0].measures[0].voices[0].beats[0]
    bend6_beat = song_bend_6.tracks[0].measures[0].voices[0].beats[0]
    bend7_beat = song_bend_7.tracks[0].measures[0].voices[0].beats[0]

    song_slide_1 = gp.parse('./gprofiles/slide_1.gp5')
    song_slide_2 = gp.parse('./gprofiles/slide_2.gp5')
    song_slide_3 = gp.parse('./gprofiles/slide_3.gp5')
    song_slide_4 = gp.parse('./gprofiles/slide_4.gp5')
    song_slide_5 = gp.parse('./gprofiles/slide_5.gp5')
    song_slide_6 = gp.parse('./gprofiles/slide_6.gp5')
    slide1_beat = song_slide_1.tracks[0].measures[0].voices[0].beats[0]
    slide2_beat = song_slide_2.tracks[0].measures[0].voices[0].beats[0]
    slide3_beat = song_slide_3.tracks[0].measures[0].voices[0].beats[0]
    slide4_beat = song_slide_4.tracks[0].measures[0].voices[0].beats[0]
    slide5_beat = song_slide_5.tracks[0].measures[0].voices[0].beats[0]
    slide6_beat = song_slide_6.tracks[0].measures[0].voices[0].beats[0]

    song_dead = gp.parse('./gprofiles/dead.gp5')
    dead_beat = song_dead.tracks[0].measures[0].voices[0].beats[0]

    song_hammer = gp.parse('./gprofiles/hammer.gp5')
    hammer_beat = song_hammer.tracks[0].measures[0].voices[0].beats[0]

    song_vibrato = gp.parse('./gprofiles/vibrato.gp5')
    vibrato_beat = song_vibrato.tracks[0].measures[0].voices[0].beats[0]

    song_harmonic_1 = gp.parse('./gprofiles/harmonic_1.gp5')
    harmonic_1_beat = song_harmonic_1.tracks[0].measures[0].voices[0].beats[0]
    # Create a new Guitar Pro song
    song = gp.models.Song()

    # Set the song's information
    song.title = titlename
    song.artist = "DjentleViBe"
    song.tempo = 120  # Set the tempo
    song.tracks[0].name = "Guitar"
    song.tracks[0].channel.instrument = 30
    
    song.tracks[0].measures[0].timeSignature.numerator = adjustmeasure(beatval)
    song.tracks[0].strings[0].value = 58
    song.tracks[0].strings[1].value = 53
    song.tracks[0].strings[2].value = 49
    song.tracks[0].strings[3].value = 44
    song.tracks[0].strings[4].value = 39
    song.tracks[0].strings[5].value = 32


    voice = song.tracks[0].measures[0].voices[0]

    k_val = 0
    l_val = 0
    beat_collect = []
    note_collect = []
    for n, note in enumerate(noteval):

        if note == EOS:
            # print("-----EOS-----")
            continue
        elif note == BOS:
            # print("-----BOS-----")
            continue
        elif note == BARRE_NOTE:
            if l_val != 0:
                l_val -= 1
        elif note == DEAD_NOTE:
            dead_beat.notes[0].value = note_collect[l_val - 1].value
            dead_beat.notes[0].string = note_collect[l_val - 1].string
            dead_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
            voice.beats[l_val - 1] = dead_beat
            continue
        elif note == HAMMER:
            hammer_beat.notes[0].value = note_collect[l_val - 1].value
            hammer_beat.notes[0].string = note_collect[l_val - 1].string
            hammer_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
            voice.beats[l_val - 1] = hammer_beat
            continue
        elif note == VIBRATO:
            vibrato_beat.notes[0].value = note_collect[l_val - 1].value
            vibrato_beat.notes[0].string = note_collect[l_val - 1].string
            vibrato_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
            voice.beats[l_val - 1] = vibrato_beat
            continue
        elif note == HARMONIC_1:
            harmonic_1_beat.notes[0].value = note_collect[l_val - 1].value
            harmonic_1_beat.notes[0].string = note_collect[l_val - 1].string
            harmonic_1_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
            voice.beats[l_val - 1] = harmonic_1_beat
            continue
            # print("-----Barred Note-----")
        elif TREM_BAR_1 <= note <= TREM_BAR_5:
            beat_collect[l_val - 1].effect.isBend = True
            
            if note == TREM_BAR_1:
                trem1_beat.notes[0].value = note_collect[l_val - 1].value
                trem1_beat.notes[0].string = note_collect[l_val - 1].string
                trem1_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = trem1_beat
            elif note == TREM_BAR_2:
                trem2_beat.notes[0].value = note_collect[l_val - 1].value
                trem2_beat.notes[0].string = note_collect[l_val - 1].string
                trem2_beat.notes[0].beat.duration.value = beat_collect[l_val - 1].duration.value
                voice.beats[l_val - 1] = trem2_beat
            elif note == TREM_BAR_3:
                beat_collect[l_val - 1].effect.tremoloBar.type = 3
            elif note == TREM_BAR_4:
                trem4_beat.notes[0].value = note_collect[l_val - 1].value
                trem4_beat.notes[0].string = note_collect[l_val - 1].string
                trem4_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = trem4_beat
            elif note == TREM_BAR_5:
                trem5_beat.notes[0].value = note_collect[l_val - 1].value
                trem5_beat.notes[0].string = note_collect[l_val - 1].string
                trem5_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = trem5_beat
            continue
            
        elif BEND_NOTE_1 <= note <= BEND_NOTE_7:
            beat_collect[l_val - 1].effect.isBend = True
            if note == BEND_NOTE_1:
                bend1_beat.notes[0].value = note_collect[l_val - 1].value
                bend1_beat.notes[0].string = note_collect[l_val - 1].string
                bend1_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = bend1_beat
            elif note == BEND_NOTE_2:
                bend2_beat.notes[0].value = note_collect[l_val - 1].value
                bend2_beat.notes[0].string = note_collect[l_val - 1].string
                bend2_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = bend2_beat
            elif note == BEND_NOTE_3:
                bend3_beat.notes[0].value = note_collect[l_val - 1].value
                bend3_beat.notes[0].string = note_collect[l_val - 1].string
                bend3_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = bend3_beat
            elif note == BEND_NOTE_4:
                bend4_beat.notes[0].value = note_collect[l_val - 1].value
                bend4_beat.notes[0].string = note_collect[l_val - 1].string
                bend4_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = bend4_beat
            elif note == BEND_NOTE_5:
                bend5_beat.notes[0].value = note_collect[l_val - 1].value
                bend5_beat.notes[0].string = note_collect[l_val - 1].string
                bend5_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = bend5_beat
            elif note == BEND_NOTE_6:
                bend6_beat.notes[0].value = note_collect[l_val - 1].value
                bend6_beat.notes[0].string = note_collect[l_val - 1].string
                bend6_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = bend6_beat
            elif note == BEND_NOTE_7:
                bend7_beat.notes[0].value = note_collect[l_val - 1].value
                bend7_beat.notes[0].string = note_collect[l_val - 1].string
                bend7_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = bend7_beat
            continue

        elif SLIDE_NOTE_1 <= note <= SLIDE_NOTE_6:
            if note == SLIDE_NOTE_1:
                slide1_beat.notes[0].value = note_collect[l_val - 1].value
                slide1_beat.notes[0].string = note_collect[l_val - 1].string
                slide1_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = slide1_beat
            elif note == SLIDE_NOTE_2:
                slide2_beat.notes[0].value = note_collect[l_val - 1].value
                slide2_beat.notes[0].string = note_collect[l_val - 1].string
                slide2_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = slide2_beat
            elif note == SLIDE_NOTE_3:
                slide3_beat.notes[0].value = note_collect[l_val - 1].value
                slide3_beat.notes[0].string = note_collect[l_val - 1].string
                slide3_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = slide3_beat
            elif note == SLIDE_NOTE_4:
                slide4_beat.notes[0].value = note_collect[l_val - 1].value
                slide4_beat.notes[0].string = note_collect[l_val - 1].string
                slide4_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = slide4_beat
            elif note == SLIDE_NOTE_5:
                slide5_beat.notes[0].value = note_collect[l_val - 1].value
                slide5_beat.notes[0].string = note_collect[l_val - 1].string
                slide5_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = slide5_beat
            elif note == SLIDE_NOTE_6:
                slide6_beat.notes[0].value = note_collect[l_val - 1].value
                slide6_beat.notes[0].string = note_collect[l_val - 1].string
                slide6_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = slide6_beat
            continue
        else:
            beat_collect.append(gp.Beat(voice=voice))
            voice.beats.append(beat_collect[k_val])
            note_collect.append(gp.Note(beat = beat_collect[k_val]))
            note_collect[l_val].value = note
            note_collect[l_val].effect.palmMute = palmval[n]
            note_collect[l_val].string = min(stringnum[n], 6)

            note_collect[l_val].beat.duration.value, b_val = getnotetype(beatval[n])

            if b_val >= 8:
                note_collect[k_val].beat.duration.isDotted = True
            if b_val in (2, 9):
                note_collect[k_val].beat.duration.tuplet.enters = 3
                note_collect[k_val].beat.duration.tuplet.times = 2
            if b_val in (3, 10):
                note_collect[k_val].beat.duration.tuplet.enters = 5
                note_collect[k_val].beat.duration.tuplet.times = 4
            if b_val in (4, 11):
                note_collect[k_val].beat.duration.tuplet.enters = 6
                note_collect[k_val].beat.duration.tuplet.times = 4
            if b_val in (5, 12):
                note_collect[k_val].beat.duration.tuplet.enters = 7
                note_collect[k_val].beat.duration.tuplet.times = 4
            if b_val in (6, 13):
                note_collect[k_val].beat.duration.tuplet.enters = 9
                note_collect[k_val].beat.duration.tuplet.times = 8
            if b_val in (7, 14):
                note_collect[k_val].beat.duration.tuplet.enters = 11
                note_collect[k_val].beat.duration.tuplet.times = 8

            beat_collect[k_val].notes.append(note_collect[l_val])
            k_val += 1
            if l_val != MAX_SEQ_LENGTH - 1:
                l_val += 1
    
    return song

def writegpro(filename, song):
    """write gpro file to disk"""
    # Save the song to a Guitar Pro file
    with open("./RESULTS/" + "/" + BACKUP + "/" + filename + ".gp5", 'wb') as file:
        gp.write(song, file)

def writebincount(counts, filename):
    # Write to a text file
    with open(filename, 'w') as file:
        for idx, count in enumerate(counts):
            file.write(f"Value {idx}: {count} occurrences\n")
    
def readbincount(filename):
    # Read the counts from the text file
    counts = []
    with open(filename, 'r') as file:
        for line in file:
            # Extract the count from each line
            parts = line.strip().split(':')
            count = int(parts[1].split()[0])  # Get the integer count
            counts.append(count)
    
    return counts

def KLDivergence(train, eval):
    N_tot = sum(train)
    Q_tot = sum(eval)
    P = []
    Q = []
    D = []
    eps = 1E-10

    for p_val in train:
        P.append(p_val / N_tot)

    for q_val in eval:
        Q.append(q_val / Q_tot)

    for item1, item2 in zip(P, Q):
        if item1 > 0:
            D.append(item1 * np.log(item1 / (item2 + eps)))
    return sum(D)

from PIL import Image
def combinepng(image1, image2, filename):

    # Load PNG images
    img1 = Image.open(image1)
    img2 = Image.open(image2)

    # Ensure both images have the same mode (e.g., RGBA) for transparency
    img1 = img1.convert("RGBA")
    img2 = img2.convert("RGBA")
    width = max(img1.width, img2.width)
    height = img1.height + img2.height
    combined_vertical = Image.new("RGBA", (width, height))

    combined_vertical.paste(img1, (0, 0))
    combined_vertical.paste(img2, (0, img1.height))

    combined_vertical.save(filename)
    print("Combined image saved vertically!")

    return 0