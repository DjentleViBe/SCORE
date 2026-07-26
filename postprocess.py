#pylint: disable=too-many-statements, too-many-branches
#pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments
"""Post processing"""
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import guitarpro as gp
import numpy as np
from matplotlib.patches import Patch
from PIL import Image
from inference import multinomial_sample, create_causal_mask
from config import (BACKUP, MAX_SEQ_LENGTH, EOS, BOS, BARRE_NOTE,
                    BEND_NOTE_1, BEND_NOTE_2, BEND_NOTE_3,
                    BEND_NOTE_4, BEND_NOTE_5, BEND_NOTE_6,
                    BEND_NOTE_7, TREM_BAR_1, TREM_BAR_2, TREM_BAR_3,
                    TREM_BAR_4, TREM_BAR_5, DEAD_NOTE, SLIDE_NOTE_1,
                    SLIDE_NOTE_2, SLIDE_NOTE_3, SLIDE_NOTE_4, SLIDE_NOTE_5,
                    SLIDE_NOTE_6, HAMMER, VIBRATO, HARMONIC_1, BAR, TUNING,
                    TEMPERATURE, TEST_CRITERIA, PREDICTION_CRITERIA, VERBOSE)

valid_tuplets = [(1, 1), (3, 2), (5, 4), (6, 4), (7, 4), (9, 8), (10, 8), (11, 8), (12, 8), (13, 8)]

def plot_multiple(plot_collect, labels, text, filename):
    "Plots data and saves figure"
    plt.suptitle('SupervisedRiffGen')
    plt.title('Loss = ' + str(round(text, 4)))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    for i, _ in enumerate(plot_collect):
        plt.plot(plot_collect[i], label=labels[i])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
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
    """
    Plot bar chart and save to file
    
    :param labels: labels for the x-axis
    :param title: title of the plot
    :param counts: bincounts
    :param filename: filename to export
    """
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts, color='black', edgecolor='black')
    plt.ylabel('Occurrences', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=90)

    for barval in bars:
        yval = barval.get_height()
        plt.text(barval.get_x() + barval.get_width()/2, yval + 0.5,
                 int(yval), ha='center', va='bottom', fontsize=7)

    plt.grid(axis='y', linestyle='--', alpha=0.7, color = 'k')
    plt.tight_layout()
    plt.savefig(filename, dpi = 200)

    plt.close()

    return 0

def plotbarlog(labels, title, counts, filename):
    """
    Plot bar chart with log and save to file
    
    :param labels: labels for the x-axis
    :param title: title of the plot
    :param counts: bincounts
    :param filename: filename to export
    """
    plt.cla()
    plt.figure(figsize=(10, 6))

    # Plot bar chart
    bars = plt.bar(range(len(labels)), counts, color='black', edgecolor='black')

    # Set custom tick positions and labels
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=90)

    # Set axis labels and title
    plt.ylabel('Occurrences', fontsize=12)
    plt.title(title, fontsize=14)

    # Add count labels above bars
    for barval in bars:
        yval = barval.get_height()
        if yval > 0:
            plt.text(barval.get_x() + barval.get_width()/2,
                     yval + 0.5, int(yval), ha='center', va='bottom')
        # Set y-axis to log scale
    plt.yscale('log')
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='k')
    plt.tight_layout()

    # Save the figure
    plt.savefig(filename, dpi=200)
    plt.close()

    return 0

def plotbar_dual(title, labels, counts1, counts2, kldiv,
                 filename, label1='Training', label2='Inference',
                 fontsize=12):
    """
    Plot bar dual chart and save to file
    
    :param labels: labels for the x-axis
    :param title: title of the plot
    :param counts1-2: bincounts
    :param filename: filename to export
    :param KLD: KL divergence values
    :param label1-2: legend labels
    """
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))  # label positions
    width = 0.4  # width of the bars

    # Primary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars1 = ax1.bar(x - width/2, counts1, width, label=label1,
                    color='black', edgecolor='black')
    # ax1.set_xlabel('Notes', fontsize=12)
    ax1.set_ylabel('Training', fontsize=12)
    ax1.tick_params(axis='y')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=fontsize)
    #fig.suptitle(title, fontsize=18)      # Title for the Axes
    ax1.set_title('KL-Divergence = ' + str(round(kldiv, 4)), fontsize = 20)
    plt.xticks(rotation=90)
    # Annotate primary bars
    for barval in bars1:
        yval = barval.get_height()
        if yval != 0:
            ax1.text(
                barval.get_x() + barval.get_width() / 2,    # horizontal center
                yval*1.02,                             # halfway inside the bar
                int(yval),                            # the label
                ha='center', va='bottom',             # centered inside
                fontsize=14,
                rotation=90,                          # rotate text
                color='black'                        # use contrasting color
            )
    # Secondary axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, counts2, width, label=label2,
                    color='lightgray', edgecolor='black')
    ax2.set_ylabel('Inference', fontsize=12)
    ax2.tick_params(axis='y')
    ax1.set_ylim(bottom=max(1, min(counts1)*0.1), top = max(counts1)*6)
    ax2.set_ylim(bottom=max(1, min(counts2)*0.1), top = max(counts2)*6)

    # Annotate secondary bars
    for barval in bars2:
        yval = barval.get_height()
        if yval != 0:
            ax2.text(
                barval.get_x() + barval.get_width() / 2,    # horizontal center
                yval*1.02,                             # halfway inside the bar
                int(yval),                            # the label
                ha='center', va='bottom',             # centered inside
                fontsize=14,
                rotation=90,                          # rotate text
                color='black'                        # use contrasting colo
            )
    # Combine legends
    labels = [label1, label2]
    legend_patches = [
        Patch(facecolor='black', edgecolor='black', label=label1),
        Patch(facecolor='lightgray', edgecolor='black', label=label2)
    ]
    ax1.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.1, 0.5))
    # Grid on primary axis
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

    return 0

def plotbar_quad(labels, counts1, counts2, kldiv, filename):
    """
    Docstring for plotbar_quad
    
    :param labels: labels for the x-axis
    :param counts1: counts for training
    :param counts2: counts for inference
    :param kldiv: KL divergence values
    :param filename: filename to export
    """
    titles = ["(a)", "(b)", "(c)", "(d)"]
    fig, ax = plt.subplots(2, 2 ,figsize=(14, 10))
    for i, l in enumerate(labels):
        x = np.arange(len(labels[i]))  # label positions
        width = 0.4  # width of the bars
        row = i // 2
        col = i % 2
        ax1 = ax[row, col]
        bars1 = ax1.bar(x - width/2, counts1[i], width, label="Training",
                        color='black', edgecolor='black')
        # ax1.set_xlabel('Notes', fontsize=12)
        ax1.set_ylabel('Training', fontsize=16)
        ax1.tick_params(axis='y')
        ax1.set_xticks(x)
        ax1.set_xticklabels(l, fontsize=11, rotation=90)
        #fig.suptitle(title, fontsize=18)      # Title for the Axes
        ax1.set_title(r"$\bf{" + titles[i] + r"}$" +
                      " KL-Divergence = " + str(round(kldiv[i], 4)),fontsize=20)# Annotate primary bars
        for barval in bars1:
            yval = barval.get_height()
            if yval != 0:
                ax1.text(
                    barval.get_x() + barval.get_width() / 2,    # horizontal center
                    yval*1.02,                             # halfway inside the bar
                    int(yval),                            # the label
                    ha='center', va='bottom',             # centered inside
                    fontsize=10,
                    rotation=90,                          # rotate text
                    color='black'                        # use contrasting color
                )
        # Secondary axis
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, counts2[i], width, label="Infrence",
                        color='lightgray', edgecolor='black')
        ax2.set_ylabel('Inference', fontsize=16)
        ax2.tick_params(axis='y')
        ax1.set_ylim(bottom=max(1, min(counts1[i])*0.1), top = max(counts1[i])*10)
        ax2.set_ylim(bottom=max(1, min(counts2[i])*0.1), top = max(counts2[i])*10)

        # Annotate secondary bars
        for barval in bars2:
            yval = barval.get_height()
            if yval != 0:
                ax2.text(
                    barval.get_x() + barval.get_width() / 2,    # horizontal center
                    yval*1.02,                             # halfway inside the bar
                    int(yval),                            # the label
                    ha='center', va='bottom',             # centered inside
                    fontsize=10,
                    rotation=90,                          # rotate text
                    color='black'                        # use contrasting colo
                )
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax1.set_xlim(left=x[0] - width - 0.01, right=x[-1] + width + 0.01)
    # Combine legends
    labels = ["Training", "Inference"]
    legend_patches = [
        Patch(facecolor='black', edgecolor='black', label="Training"),
        Patch(facecolor='lightgray', edgecolor='black', label="Inference")
    ]
    fig.legend(handles=legend_patches, loc="lower center",
            ncol=len(legend_patches),frameon=False,
            fontsize=16)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(filename, dpi=200)
    plt.close()


def decoder_search(decoder, dummy_in, embedding_layer, pos_enc, mask, e_val):
    """
    Transformer Decoder Search Step
    
    :param decoder: decoder model
    :param dummy_in: output from previous step
    :param embedding_layer: embedding layer
    :param pos_enc: positional encoding
    :param mask: masking
    :param e_val: current evaluation step
    """
    embeddings = embedding_layer(dummy_in[:, :e_val])
    pos_enc = pos_enc.unsqueeze(0)
    positional = pos_enc[:, :e_val, :]
    output_eval, _, _ = decoder(embeddings + positional, mask)

    next_token_logits = output_eval[:, -1, :]
    scaled_logits = next_token_logits / TEMPERATURE

    probabilities = F.softmax(scaled_logits, dim=-1)
    next_token = None
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

def decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask, seq_lim, device):
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
                    next_token = decoder_search(decoder, beam, embedding_layer,
                                                pos_enc, mask, e_val)

                    # Extend each beam with the top k candidates
                    for bw in range(beam_width):
                        # Copy the current beam
                        new_beam = beam.clone()
                        # Update the token for this position
                        new_beam[0][e_val] = next_token[0][bw]
                        # Update the score (log probability)
                        new_score = beam_scores[i] + next_token[0][bw].log()
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
                dummy_in[0][1:] = EOS
                for e_val in range (1, seq_lim):
                    casualmask = create_causal_mask(e_val, device)
                    next_token = decoder_search(decoder, dummy_in, embedding_layer,
                                                pos_enc, casualmask, e_val)
                    # generated_sequence = dummy_in
                    if next_token == EOS:
                        print("End detected")
                        break
                    dummy_in[0][e_val] = next_token
            else:
                e_val = 2
                trial = 0
                temperature_var = TEMPERATURE
                while e_val < seq_lim:
                    next_token = decoder_search(decoder, dummy_in, embedding_layer,
                                                pos_enc, mask, e_val)

                    if e_val != 2:
                        if dummy_in[0][e_val - 1] < BOS < next_token:
                            dummy_in[0][e_val] = next_token
                            e_val +=1
                        elif next_token < BOS:
                            dummy_in[0][e_val] = next_token
                            e_val +=1
                        else:
                            temperature_var /= 2
                            print("Trial -->", e_val, trial, next_token.cpu()[0][0],
                                  dummy_in.cpu()[0][e_val - 1])
                            trial += 1
                    else:
                        # generated_sequence = dummy_in
                        dummy_in[0][e_val] = next_token
                        e_val += 1

        print(dummy_in)
    return dummy_in

def check_measure(duration_sum, n):
    """Check if measure exceeds 4/4 time signature"""
    if math.ceil(duration_sum) > 32:
        print("Exiting early at Token : ", n + 1)
        print("total duration : ", math.ceil(duration_sum))
        return True
    return False

def findnewstring(note, string):
    """
    Find new string for note in chord
    
    :param note: note value
    :param string: string number
    """
    notepos = TUNING[string - 1] + note
    if string > 4:
        notepos += 12
    else:
        notepos -= 12
    closest_index = min(range(len(TUNING)), key=lambda i: abs(TUNING[i] - notepos))
    return max(0, notepos - TUNING[closest_index]), closest_index

def makegpro(titlename, noteval, stringnum, beatval, palmval):
    """Generate gpro file"""
    # read bend and tremolo templates
    song_trem_1 = gp.parse('./gprofiles/gp5_templates/trem_1.gp5')
    song_trem_2 = gp.parse('./gprofiles/gp5_templates/trem_2.gp5')
    song_trem_4 = gp.parse('./gprofiles/gp5_templates/trem_4.gp5')
    song_trem_3 = gp.parse('./gprofiles/gp5_templates/trem_3.gp5')
    song_trem_5 = gp.parse('./gprofiles/gp5_templates/trem_5.gp5')
    trem1_beat = song_trem_1.tracks[0].measures[0].voices[0].beats[0]
    trem2_beat = song_trem_2.tracks[0].measures[0].voices[0].beats[0]
    trem3_beat = song_trem_3.tracks[0].measures[0].voices[0].beats[0]
    trem4_beat = song_trem_4.tracks[0].measures[0].voices[0].beats[0]
    trem5_beat = song_trem_5.tracks[0].measures[0].voices[0].beats[0]

    song_bend_1 = gp.parse('./gprofiles/gp5_templates/bend_1.gp5')
    song_bend_2 = gp.parse('./gprofiles/gp5_templates/bend_2.gp5')
    song_bend_3 = gp.parse('./gprofiles/gp5_templates/bend_3.gp5')
    song_bend_4 = gp.parse('./gprofiles/gp5_templates/bend_4.gp5')
    song_bend_5 = gp.parse('./gprofiles/gp5_templates/bend_5.gp5')
    song_bend_6 = gp.parse('./gprofiles/gp5_templates/bend_6.gp5')
    song_bend_7 = gp.parse('./gprofiles/gp5_templates/bend_7.gp5')
    bend1_beat = song_bend_1.tracks[0].measures[0].voices[0].beats[0]
    bend2_beat = song_bend_2.tracks[0].measures[0].voices[0].beats[0]
    bend3_beat = song_bend_3.tracks[0].measures[0].voices[0].beats[0]
    bend4_beat = song_bend_4.tracks[0].measures[0].voices[0].beats[0]
    bend5_beat = song_bend_5.tracks[0].measures[0].voices[0].beats[0]
    bend6_beat = song_bend_6.tracks[0].measures[0].voices[0].beats[0]
    bend7_beat = song_bend_7.tracks[0].measures[0].voices[0].beats[0]

    song_slide_1 = gp.parse('./gprofiles/gp5_templates/slide_1.gp5')
    song_slide_2 = gp.parse('./gprofiles/gp5_templates/slide_2.gp5')
    song_slide_3 = gp.parse('./gprofiles/gp5_templates/slide_3.gp5')
    song_slide_4 = gp.parse('./gprofiles/gp5_templates/slide_4.gp5')
    song_slide_5 = gp.parse('./gprofiles/gp5_templates/slide_5.gp5')
    song_slide_6 = gp.parse('./gprofiles/gp5_templates/slide_6.gp5')
    slide1_beat = song_slide_1.tracks[0].measures[0].voices[0].beats[0]
    slide2_beat = song_slide_2.tracks[0].measures[0].voices[0].beats[0]
    slide3_beat = song_slide_3.tracks[0].measures[0].voices[0].beats[0]
    slide4_beat = song_slide_4.tracks[0].measures[0].voices[0].beats[0]
    slide5_beat = song_slide_5.tracks[0].measures[0].voices[0].beats[0]
    slide6_beat = song_slide_6.tracks[0].measures[0].voices[0].beats[0]

    song_dead = gp.parse('./gprofiles/gp5_templates/dead.gp5')
    dead_beat = song_dead.tracks[0].measures[0].voices[0].beats[0]

    song_hammer = gp.parse('./gprofiles/gp5_templates/hammer.gp5')
    hammer_beat = song_hammer.tracks[0].measures[0].voices[0].beats[0]

    song_vibrato = gp.parse('./gprofiles/gp5_templates/vibrato.gp5')
    vibrato_beat = song_vibrato.tracks[0].measures[0].voices[0].beats[0]

    song_harmonic_1 = gp.parse('./gprofiles/gp5_templates/harmonic_1.gp5')
    harmonic_1_beat = song_harmonic_1.tracks[0].measures[0].voices[0].beats[0]
    # Create a new Guitar Pro song
    song = gp.models.Song()

    # Set the song's information
    song.title = titlename
    song.artist = "DjentleViBe"
    song.tempo = 120  # Set the tempo
    song.tracks[0].name = "Guitar"
    song.tracks[0].channel.instrument = 30

    song.tracks[0].measures[0].hasTimeSignature  = True
    song.tracks[0].measures[0].timeSignature.denominator.value = 4
    voice = song.tracks[0].measures[0].voices[0]
    k_val = 0
    l_val = 0
    beat_collect = []
    note_collect = []
    duration_sum = 0
    reuse_last_beat = False
    tripletcount = 0
    quintupletcount = 0
    sextupletcount = 0
    septupletcount = 0
    ninetupletcount = 0
    eleventupletcount = 0
    last_beat_duration = 0
    tuplet_on = False
    enters = None
    times = None
    for n, note in enumerate(noteval):
        if not tuplet_on:
            enters = None
            times = None
        if note in (EOS, BOS, BAR):
            continue
        if n == 0:
            beatval[n]["tuplet"] = 3
            beatval[n]["times"] = 2
        if n == 1:
            note = BARRE_NOTE
        #if n >= 15:
        #    continue
        # print(n, k_val, reuse_last_beat, note)
        if beatval[n]["tuplet"] == 3 and tripletcount == 0 and not tuplet_on:
            tripletcount = 1
            last_beat_duration = beatval[n]["duration"]
            tuplet_on = True
        elif beatval[n]["tuplet"] == 5 and quintupletcount == 0 and not tuplet_on:
            quintupletcount = 1
            last_beat_duration = beatval[n]["duration"] 
            tuplet_on = True
        elif beatval[n]["tuplet"] == 6 and sextupletcount == 0 and not tuplet_on:
            sextupletcount = 1
            last_beat_duration = beatval[n]["duration"]
            tuplet_on = True
        elif beatval[n]["tuplet"] == 7 and septupletcount == 0 and not tuplet_on:
            septupletcount = 1
            last_beat_duration = beatval[n]["duration"]
            tuplet_on = True
        elif beatval[n]["tuplet"] == 9 and ninetupletcount == 0 and not tuplet_on:
            ninetupletcount = 1
            last_beat_duration = beatval[n]["duration"]
            tuplet_on = True
        elif beatval[n]["tuplet"] == 11 and eleventupletcount == 0 and not tuplet_on:
            eleventupletcount = 1
            last_beat_duration = beatval[n]["duration"]
            tuplet_on = True
        # print("beatval", beatval[n]["tuplet"])
        dotted = False
        base_duration = 0
        if beatval[n]["dotted"] == True and not tuplet_on:
            note_collect[l_val].beat.duration.isDotted = True
            dotted = True
        if tuplet_on:
            beatval[n]["duration"] = last_beat_duration
        if note == BARRE_NOTE:
            if k_val != 0:
                reuse_last_beat = True
                reused_beat = beat_collect[k_val - 1]
            continue
        elif note == DEAD_NOTE:
            reuse_last_beat = False
            if note_collect and noteval[n - 1] < BAR:
                note_collect[l_val - 1].type = dead_beat.notes[0].type
                continue
        elif note == HAMMER:
            reuse_last_beat = False
            if note_collect and noteval[n - 1] < BAR:
                note_collect[l_val - 1].effect.hammer = hammer_beat.notes[0].effect.hammer
                continue
        elif note == VIBRATO:
            reuse_last_beat = False
            if note_collect and noteval[n - 1] < BAR:
                note_collect[l_val - 1].effect.vibrato = vibrato_beat.notes[0].effect.vibrato
                continue
        elif note == HARMONIC_1:
            reuse_last_beat = False
            if note_collect and noteval[n - 1] < BAR:
                note_collect[l_val - 1].effect.harmonic = harmonic_1_beat.notes[0].effect.harmonic
                continue
        elif TREM_BAR_1 <= note <= TREM_BAR_5:
            reuse_last_beat = False
            if note_collect and noteval[n - 1] < BAR:
                if note == TREM_BAR_1:
                    beat_collect[k_val - 1].effect.tremoloBar = trem1_beat.notes[0].beat.effect.tremoloBar
                elif note == TREM_BAR_2:
                    beat_collect[k_val - 1].effect.tremoloBar = trem2_beat.notes[0].beat.effect.tremoloBar
                elif note == TREM_BAR_3:
                    beat_collect[k_val - 1].effect.tremoloBar = trem3_beat.notes[0].beat.effect.tremoloBar
                elif note == TREM_BAR_4:
                    beat_collect[k_val - 1].effect.tremoloBar = trem4_beat.notes[0].beat.effect.tremoloBar
                elif note == TREM_BAR_5:
                    beat_collect[k_val - 1].effect.tremoloBar = trem5_beat.notes[0].beat.effect.tremoloBar
                note_collect[l_val - 1].effect.isTremoloBar = True
                continue
            
        elif BEND_NOTE_1 <= note <= BEND_NOTE_7:
            reuse_last_beat = False
            if note_collect and noteval[n - 1] < BAR:
                if k_val != 0:
                    beat_collect[k_val - 1].effect.isBend = True
                if note == BEND_NOTE_1:
                    note_collect[l_val - 1].effect.bend = bend1_beat.notes[0].effect.bend
                elif note == BEND_NOTE_2:
                    note_collect[l_val - 1].effect.bend = bend2_beat.notes[0].effect.bend
                elif note == BEND_NOTE_3:
                    note_collect[l_val - 1].effect.bend = bend3_beat.notes[0].effect.bend
                elif note == BEND_NOTE_4:
                    note_collect[l_val - 1].effect.bend = bend4_beat.notes[0].effect.bend
                elif note == BEND_NOTE_5:
                    note_collect[l_val - 1].effect.bend = bend5_beat.notes[0].effect.bend
                elif note == BEND_NOTE_6:
                    note_collect[l_val - 1].effect.bend = bend6_beat.notes[0].effect.bend
                elif note == BEND_NOTE_7:
                    note_collect[l_val - 1].effect.bend = bend7_beat.notes[0].effect.bend
                continue

        elif SLIDE_NOTE_1 <= note <= SLIDE_NOTE_6:
            reuse_last_beat = False
            if note_collect and noteval[n - 1] < BAR:
                if note == SLIDE_NOTE_1:
                    note_collect[l_val - 1].effect.slides = slide1_beat.notes[0].effect.slides
                elif note == SLIDE_NOTE_2:
                    note_collect[l_val - 1].effect.slides = slide2_beat.notes[0].effect.slides
                elif note == SLIDE_NOTE_3:
                    note_collect[l_val - 1].effect.slides = slide3_beat.notes[0].effect.slides
                elif note == SLIDE_NOTE_4:
                    note_collect[l_val - 1].effect.slides = slide4_beat.notes[0].effect.slides
                elif note == SLIDE_NOTE_5:
                    note_collect[l_val - 1].effect.slides = slide5_beat.notes[0].effect.slides
                elif note == SLIDE_NOTE_6:
                    note_collect[l_val - 1].effect.slides = slide6_beat.notes[0].effect.slides
                continue
        else:
            if reuse_last_beat:  # e.g. from BARRE_NOTE
                # check if the note is the same
                if VERBOSE == 1:
                    print("chord", note, stringnum[n])
                used_strings = {note.string for note in reused_beat.notes}
                # Collect all strings already used in the reused beat
                # If the current string is already used in this chord, find a new one
                if stringnum[n] in used_strings:
                    if VERBOSE == 1:
                        print(f"String {stringnum[n]} already used in chord — finding alternate for note {noteval[n]}")
                    noteval[n], stringnum[n] = findnewstring(noteval[n], stringnum[n])

                # Re-check after changing to avoid accidental collisions again (optional safeguard)
                if stringnum[n] in used_strings:
                    print("Used strings:", used_strings)
                    print("All candidate strings:", stringnum)
                    print("Current note:", noteval[n])
                    continue  # skip assignment for rests
                current_beat = reused_beat
                if tuplet_on:
                    current_beat.duration.value = last_beat_duration
                    current_beat.duration.tuplet.enters = enters
                    current_beat.duration.tuplet.times = times
                reuse_last_beat = False
            else:
                # Duration comes from beatval[n]
                current_beat = gp.Beat(voice=voice)
                if beatval[n].get("dotted") and not tuplet_on:
                    base_duration *= 1.5
                # logic to check if the subdivision condition is met
                if tripletcount > 0:
                    beatval[n]["duration"] = last_beat_duration
                    enters = 3
                    times = 2
                    tripletcount += 1
                    if n < MAX_SEQ_LENGTH - 1:
                        if tripletcount == 4 and not reuse_last_beat and noteval[n + 1] != BARRE_NOTE:
                            tripletcount = 0
                            tuplet_on = False
                elif quintupletcount > 0:
                    beatval[n]["duration"] = last_beat_duration
                    enters = 5
                    times = 4
                    quintupletcount += 1
                    if n < MAX_SEQ_LENGTH - 1:
                        if quintupletcount == 6 and not reuse_last_beat and noteval[n + 1] != BARRE_NOTE:
                            quintupletcount = 0
                            tuplet_on = False
                elif sextupletcount > 0:
                    beatval[n]["duration"] = last_beat_duration
                    enters = 6
                    times = 4
                    sextupletcount += 1
                    if n < MAX_SEQ_LENGTH - 1:
                        if sextupletcount == 7 and not reuse_last_beat and noteval[n + 1] != BARRE_NOTE:
                            sextupletcount = 0
                            tuplet_on = False
                elif septupletcount > 0:
                    beatval[n]["duration"] = last_beat_duration
                    enters = 7
                    times = 4
                    septupletcount += 1
                    if n < MAX_SEQ_LENGTH - 1:
                        if septupletcount == 8 and not reuse_last_beat and noteval[n + 1] != BARRE_NOTE:
                            septupletcount = 0
                            tuplet_on = False
                elif ninetupletcount > 0:
                    beatval[n]["duration"] = last_beat_duration
                    enters = 9
                    times = 8
                    ninetupletcount += 1
                    if n < MAX_SEQ_LENGTH - 1:
                        if ninetupletcount == 10 and not reuse_last_beat and noteval[n + 1] != BARRE_NOTE:
                            ninetupletcount = 0
                            tuplet_on = False
                elif eleventupletcount > 0:
                    beatval[n]["duration"] = last_beat_duration
                    enters = 11
                    times = 8
                    eleventupletcount += 1
                    if n < MAX_SEQ_LENGTH - 1:
                        if eleventupletcount == 12 and not reuse_last_beat and noteval[n + 1] != BARRE_NOTE:
                            eleventupletcount = 0
                            tuplet_on = False
                else:
                    if not tuplet_on:
                        enters = None
                        times = None
                base_duration = 4.0 / beatval[n]["duration"]
                if (enters, times) in valid_tuplets:
                    base_duration *= times / enters
                    # continue  # Skip this note if the tuplet combination is invalid
                duration_sum += base_duration
                k_val += 1
                voice.beats.append(current_beat)
                beat_collect.append(current_beat)
            # print(enters, times, beatval[n]["duration"], last_beat_duration, n, tuplet_on)
            current_beat.status = gp.models.BeatStatus.normal
            # print(note,  max(stringnum[n], 1), k_val)
            note_collect.append(gp.Note(beat=current_beat))
            note_collect[l_val].type = gp.models.NoteType.normal
            note_collect[l_val].value = note
            note_collect[l_val].effect.palmMute = palmval[n]
            note_collect[l_val].string = max(stringnum[n], 1)
            note_collect[l_val].beat.duration.value = beatval[n]["duration"]

            if (enters, times) in valid_tuplets:
                note_collect[l_val].beat.duration.tuplet.enters = enters
                note_collect[l_val].beat.duration.tuplet.times = times

            current_beat.notes.append(note_collect[l_val])
            if l_val != MAX_SEQ_LENGTH - 1:
                l_val += 1

            if check_measure(duration_sum, n):
                song.tracks[0].measures[0].timeSignature.numerator = min(32, math.ceil(duration_sum))
                return song
        if VERBOSE == 1:
            print(f"Beat : {beatval[n].get("duration")}, Tuplet : {enters}, Dotted : {dotted}, Duration_sum : {round(duration_sum, 2)}")
    # print("total duration : ", math.ceil(duration_sum))
    song.tracks[0].measures[0].timeSignature.numerator = min(32, math.ceil(duration_sum))
    return song

def writegpro(filename, song):
    """write gpro file to disk"""
    # Save the song to a Guitar Pro file
    with open("./RESULTS/" + "/" + BACKUP + "/gp5/" + filename + ".gp5", 'wb') as file:
        gp.write(song, file)

def writebincount(counts, filename):
    """Write to a text file"""
    with open(filename, 'w', encoding='utf-8') as file:
        for idx, count in enumerate(counts):
            file.write(f"Value {idx}: {count} occurrences\n")

def writebincount2(labels, counts, filename):
    """Write to a text file"""
    with open(filename, 'w', encoding='utf-8') as file:
        for idx, count in enumerate(counts):
            file.write(f"Value {labels[idx]}: {count} occurrences\n")

def readbincount(filename):
    """Read the counts from the text file"""
    counts = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # Extract the count from each line
            parts = line.strip().split(':')
            count = int(parts[1].split()[0])  # Get the integer count
            counts.append(count)

    return counts

def kldivergence(train, evaluation):
    """
    Compute Kullback-Leibler divergence between two distributions
    
    :param train: Training distribution
    :param eval: Evaluation distribution
    """
    n_tot = sum(train)
    q_tot = sum(evaluation)
    p_val = []
    q_val = []
    d_val = []
    eps = 1E-10

    for p_value in train:
        p_val.append(p_value / n_tot)

    for q_value in evaluation:
        q_val.append(q_value / q_tot)

    for item1, item2 in zip(p_val, q_val):
        if item1 > 0:
            d_val.append(item1 * np.log(item1 / (item2 + eps)))
    return sum(d_val)

def combinepng(image1, image2, filename):
    """
    Combine two PNG images vertically
    
    :param image1: First image path
    :param image2: Second image path
    :param filename: Output filename
    """
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
