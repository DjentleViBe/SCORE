"""Preprocess data required for training"""
import os
import guitarpro
import torch
import numpy as np

def readgpro(filename):
    """read gpro file"""
    song = guitarpro.parse(filename)
    return song

def getnumstrings(ginfo):
    """get string count"""
    stringcount = len(ginfo.strings)
    # print("Number of strings : ", stringcount)
    return stringcount

def gettuning(strings):
    """get tuning config"""
    return strings.value

def guitarinfo(song):
    """accumulate guitar info"""
    tuning = []
    sc_val = getnumstrings(song.tracks[0])
    for i in range(0, sc_val):
        tuning.append(gettuning(song.tracks[0].strings[i]))

    return tuning

def get_positional_encoding(seq_len, d_model):
    """Compute positional encoding"""
    positional_encoding = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
            positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i)/d_model)))

    return torch.tensor(positional_encoding, dtype=torch.float32)

def create_dir(directory_path):
    """Create directory if it does not exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    return 0
