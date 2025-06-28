"""Decoding results"""
from config import EOS, BOS, BARRE_NOTE, BEND_NOTE_1, BEND_NOTE_7, \
                    TREM_BAR_1, TREM_BAR_5, DEAD_NOTE, SLIDE_NOTE_1, SLIDE_NOTE_6, \
                    HAMMER, VIBRATO, HARMONIC_1, VERBOSE, BAR
import numpy as np

BASE_BEAT_START = {
    0 : 64,
    1 : 32,
    2 : 16,
    3 : 8,
    4 : 4,
    5 : 2,
    6 : 1,
}

TUPLET_REV = {
    0: None,
    1: 3,
    2: 5,
    3: 6,
    4: 7,
    5: 9,
    6: 11
}

def demapping_beat(beatkind):
    """returns beat type"""
    dotted = False
    if beatkind > 7:
        beatkind -= 7
        dotted = True

    return TUPLET_REV.get(beatkind), dotted

def detokenizer_1(dummy):
    """Derive notes from token"""
    note_val = 0
    note_type = 0
    string_num = 0
    palm_mute = False
    beat_type = {"duration": 0,      # e.g., 4 for quarter
                "tuplet": None,     # e.g., 3, 5, 6, etc.
                "dotted": False}
    beat_kind = 0
    if dummy == EOS:
        if VERBOSE == 1:
            print(f"-------EOS-------")
        note_val = EOS
    elif dummy == BAR:
        if VERBOSE == 1:
            print(f"-------BAR-------")
        note_val = BAR
    elif dummy == BOS:
        if VERBOSE == 1:
            print(f"-------BOS-------")
        note_val = BOS
    elif dummy == BARRE_NOTE:
        if VERBOSE == 1:
            print("--------Barred Note--------")
        note_val = BARRE_NOTE
    elif BEND_NOTE_7 >= dummy >= BEND_NOTE_1:
        if VERBOSE == 1:
            print("-------Accent_Bend----------")
        note_val = dummy
    elif TREM_BAR_5 >= dummy >= TREM_BAR_1:
        if VERBOSE == 1:
            print("-------Accent_Trem----------")
        note_val = dummy
    elif SLIDE_NOTE_6 >= dummy >= SLIDE_NOTE_1:
        if VERBOSE == 1:
            print("-------Accent_Slide----------")
        note_val = dummy
    elif dummy == DEAD_NOTE:
        if VERBOSE == 1:
            print("-------Dead note----------")
        note_val = dummy
    elif dummy == HAMMER:
        if VERBOSE == 1:
            print("-------Hammer----------")
        note_val = dummy
    elif dummy == VIBRATO:
        if VERBOSE == 1:
            print("-------Vibrato----------")
        note_val = dummy
    elif dummy == HARMONIC_1:
        if VERBOSE == 1:
            print("-------Harmonic----------")
        note_val = dummy
    else:
        if(dummy < BAR):
            palm_mute = False
            note_type = dummy % 276
            
            beat_kind = (dummy // 276) % 14
            base_beat = BASE_BEAT_START.get((dummy // (276 * 14)) % 7)
            if(base_beat == 0):
                print("caution")
            bt_0, bt_1 = demapping_beat(beat_kind)
            beat_type = {"duration": base_beat,      # e.g., 4 for quarter
                        "tuplet": bt_0,     # e.g., 3, 5, 6, etc.
                        "dotted": bt_1}
            
            if note_type > 138:
                # Palm mute
                note_type -= 138
                palm_mute = True
                string_num = (note_type) // 23 + 1
                note_val = (note_type) % string_num
            else:
                string_num = (note_type) // 23 + 1
                note_val = (note_type) % string_num
 
        if VERBOSE == 1:
            beat_type_cleaned = {k: int(v) if isinstance(v, np.integer) else v for k, v in beat_type.items()}
            print(f"Beat : {beat_type_cleaned} String : {string_num} Note : {note_val} PalmMute : {palm_mute}")


    return note_val, note_type, string_num, beat_type, palm_mute, beat_kind
