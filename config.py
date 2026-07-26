#pylint: disable
"""Defines parameters required for training and inference"""
TUNING              = [53, 48, 44, 39, 34, 27]
################################ transformers #################################
# 0: train,
# 1 : eval
# 2 : both
# 3 : load from file
MODE                =   2
BACKUP				 = "CB_Large_5"
SAVE				 = "CB_L"
BOS_TRUE            =   0
EOS_TRUE            =   1
# 0 - Vanilla
# 1 - Disregard any note > BARRE_NOTE
# 2 - n bars of generation, n = TEST_TRIES
# 3 - Same as above + last note of every bar is fed to firt note of next bar
# 4 - Same as above + loop until no repeatitive notes > BOS
TEST_CRITERIA       =   3
TEST_TRIES          =   12
# Total number of start_id to try with testing
TEST_NUM            =   3
# 1 - Greedy Search
# 2 - Beam Search
# 3 - Multinomial Sampling
# 4 - Multinomial Sampling - manual
PREDICTION_CRITERIA =   3
INFERENCE_TYPE = 0
# 0 - Vanilla
# 1 - Tuplet friendly
BEAT_ONLY = True
########## Params ##############
EPOCHS          =   125
SAVE_EVERY      =   20
VOCAB_SIZE      =   27076
FFN_HIDDEN      =   1024
MAX_SEQ_LENGTH  =   32
NUM_HEADS       =   8
DROP_PROB       =   0.3
NUM_LAYERS      =   6
D_MODEL         =   256
SCHEDULER       =   1
SCHEDULER_SIZE  =   30
LEARNING_RATE   =   0.0001
SMOOTHING       =   0.1
PATCH           =   1
STRIDE          =   1
TRAINING        =   ["CB"]
NUM_SEQUENCE    =   6208
CONVERGENCE     =   0.0005
TEMPERATURE		=   1.0
BATCH           =   32
OVERLAP         =   8
LAMBDA          =   1.5 # REPETITION PENALTY WEIGHT
ALPHA           =   0.1 # KL BIAS WEIGHT
DELTA           =   1.0 # SEQUENCE EMBEDDING PENALTY WEIGHT
NGRAM_SIZE      =   3
####################### TOKEN IDS ########################
BAR             =   27049
BOS             =   27050
BARRE_NOTE      =   27051
BEND_NOTE_1     =   27052
BEND_NOTE_2     =   27053
BEND_NOTE_3     =   27054
BEND_NOTE_4     =   27055
BEND_NOTE_5     =   27056
BEND_NOTE_6     =   27057
BEND_NOTE_7     =   27058
TREM_BAR_1      =   27059
TREM_BAR_2      =   27060
TREM_BAR_3      =   27061
TREM_BAR_4      =   27062
TREM_BAR_5      =   27063
DEAD_NOTE       =   27064
SLIDE_NOTE_1    =   27065
SLIDE_NOTE_2    =   27066
SLIDE_NOTE_3    =   27068
SLIDE_NOTE_4    =   27069
SLIDE_NOTE_5    =   27070
SLIDE_NOTE_6    =   27071
HAMMER          =   27072
VIBRATO         =   27073
HARMONIC_1      =   27074
EOS             =   27075

#################################################################################
MEASURE         =   8
VERBOSE         =   0
#################################################################################
labelsnotes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
labelsbeats = ['Base',
'Triplet',
'Quintuplet',
'Sextuplet',
'Septuplet',
'9_Tuplets',
'11_Tuplets',
'D - Base',
'D - Triplet',
'D - Quintuplet',
'D - Sextuplet',
'D - Septuplet',
'D - 9_Tuplets',
'D - 11_Tuplets']
labelsbeattype = ['1', '2', '4', '8', '16', '32', '64']
labelsaccents = ['BARRE_NOTE', 'BEND_NOTE_1', 'BEND_NOTE_2', 'BEND_NOTE_3', 'BEND_NOTE_4',
                 'BEND_NOTE_5', 'BEND_NOTE_6', 'BEND_NOTE_7', 
                    'TREM_BAR_1', 'TREM_BAR_2', 'TREM_BAR_3', 'TREM_BAR_4', 'TREM_BAR_5', 
                    'SLIDE_NOTE_1', 'SLIDE_NOTE_2', 'SLIDE_NOTE_3', 'SLIDE_NOTE_4', 
                    'SLIDE_NOTE_5', 'SLIDE_NOTE_6',
                    'DEAD_NOTE', 'HAMMER', 'VIBRATO', 'HARMONIC_1', 'PALM_MUTE']
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
PENALIZE_TOKENS = [BEND_NOTE_1, BEND_NOTE_2, BEND_NOTE_3,
                    BEND_NOTE_4,
                    BEND_NOTE_5, BEND_NOTE_6, BEND_NOTE_7,
                    TREM_BAR_1, TREM_BAR_2, TREM_BAR_3,
                    TREM_BAR_4, TREM_BAR_5,
                    DEAD_NOTE, SLIDE_NOTE_1, SLIDE_NOTE_2,
                    SLIDE_NOTE_3,
                    SLIDE_NOTE_4, SLIDE_NOTE_5, SLIDE_NOTE_6,
                    BOS, VIBRATO, HAMMER, HARMONIC_1]
