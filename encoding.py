"""Encoding gpro files"""
# Note
# C0 = 0 in GPro
from config import EOS
import config as cfg

BASE_INDEX = {
    64: 0,
    32: 1,
    16: 2,
    8 : 3,
    4 : 4,
    2 : 5,
    1 : 6
}

TUPLET_OFFSET = {
    3: 1,
    5: 2,
    6: 3,
    7: 4,
    9: 5,
    11: 6
}

MAPPING_BEAT = {
    # pure : base, triplet, quintuplet, sextuplet, septuplet, 9 tuplets, 11-tuplets (1-7)
    # dotted : base, triplet, quintuplet, sextuplet, septuplet, 9 tuplets, 11-tuplets (8-14)
    # full
    1  : 84, # 84 - 98
    # half
    2  : 70,
    # quarter
    4  : 56,
    # 8th
    8  : 42,
    # 16th
    16 : 28,
    # 32nd
    32 : 14,  # 1,2,3,4,5,6,7,8,9,10,11,12,13,14
    # 64th
    64 : 0   # 0,1,2,3,4,5,6,7,8,9,10,11,12,13
}

def tokenizer_1(note, string, duration, n_val):
    """Tokenizer type 1"""
    # 6 strings, 23 notes per string, + Palm_Mute
    # 138 * 2 total notes / beat
    # Note formula = n * 23 * String + note, n = 2 if Palm_Mute, else n = 1
    # String formula = (Note % 23) - 1
    # 82 total beats
    # 138 * 2 * 82 = 23184 unique combinations
    # 1 - 276 =  base note
    # 277 - 552 = triplet
    # .....
    # Unique combination formula = Note formula + (m * 276), m = mapping_beat number
    note_formula = (note + 1) + (string - 1) * 23 + (6 * 23) * (n_val - 1)
    beat_type = TUPLET_OFFSET.get(duration.tuplet.enters, 0)
    base_beat = BASE_INDEX[duration.value]
    if duration.isDotted:
        beat_type += 7
    encoding = note_formula + (beat_type * 276) + (base_beat * 276 * 14)
    return encoding

def beat_prob(duration):
    """
    Docstring for beat_prob
    
    :param duration: Beat duration
    """
    beatvalue = 1
    if duration.tuplet.enters == 3:
        beatvalue += 1
    if duration.tuplet.enters == 5:
        beatvalue += 2
    if duration.tuplet.enters == 6:
        beatvalue += 3
    if duration.tuplet.enters == 7:
        beatvalue += 4
    if duration.tuplet.enters == 9:
        beatvalue += 5
    if duration.tuplet.enters == 11:
        beatvalue += 6
    if duration.isDotted:
        beatvalue += 7

    return beatvalue

def beattype_prob(duration):
    """
    Docstring for beattype_prob
    
    :param duration: beat duration
    """
    beattypevalue = 1
    if duration == 1:
        beattypevalue = 1
    elif duration == 2:
        beattypevalue = 2
    elif duration == 4:
        beattypevalue = 3
    elif duration == 8:
        beattypevalue = 4
    elif duration == 16:
        beattypevalue = 5
    elif duration == 32:
        beattypevalue = 6
    elif duration == 64:
        beattypevalue = 7
    return beattypevalue

def note_prob(note, string):
    """
    Docstring for note_prob
    
    :param note: note value
    :param string: string number
    # tuning D#
    # 6: D#
    # 5: A#
    # 4: D#
    # 3: G#
    # 2: C
    # 1: F
    # Notes = C C# D D# E F F# G G# A A# B
    """
    if note > EOS:
        return 100
    if string > 6:
        print("String error : ", string)
        string -= 1
    return (note + (cfg.TUNING[string - 1] % 12)) % 12 + 1
