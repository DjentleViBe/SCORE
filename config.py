"""Defines parameters required for training and inference"""

################################ transformers #################################
# 0: train, 
# 1 : eval
# 2 : both 
# 3 : load from file
MODE                =   1
BACKUP              =   "dec_only_notes_CB_Small"
SAVE                =   "dec_only_notes_CB_Small_A"
START_ID            =   9437
BOS_TRUE            =   0
EOS_TRUE            =   1
# 0 - Vanilla
# 1 - Disregard any note > BARRE_NOTE
# 2 - n bars of generation, n = TEST_TRIES
# 3 - Same as above + last note of every bar is fed to firt note of next bar
# 4 - Same as above + loop until no repeatitive notes > BOS 
TEST_CRITERIA       =   3
TEST_TRIES          =   5
# 1 - Greedy Search
# 2 - Beam Search
# 3 - Multinomial Sampling
# 4 - Multinomial Sampling - manual
PREDICTION_CRITERIA =   3
########## Params ##############
EPOCHS          =   2
SAVE_EVERY      =   100
VOCAB_SIZE      =   23210
FFN_HIDDEN      =   512
MAX_SEQ_LENGTH  =   32
NUM_HEADS       =   4
DROP_PROB       =   0.2
NUM_LAYERS      =   6
D_MODEL         =   128
SCHEDULER       =   1
SCHEDULER_SIZE  =   30
LEARNING_RATE   =   0.00005
SMOOTHING       =   0.1
PATCH           =   1
STRIDE          =   1
TRAINING        =   ["CB"]
NUM_SEQUENCE    =   3104
CONVERGENCE     =   0.0005
TEMPERATURE     =   0.1
BATCH           =   16

EOS             =   23185
BOS             =   23186
BARRE_NOTE      =   23187
BEND_NOTE_1     =   23188
BEND_NOTE_2     =   23189
BEND_NOTE_3     =   23190
BEND_NOTE_4     =   23191
BEND_NOTE_5     =   23192
BEND_NOTE_6     =   23193
BEND_NOTE_7     =   23194
TREM_BAR_1      =   23195
TREM_BAR_2      =   23196
TREM_BAR_3      =   23197
TREM_BAR_4      =   23198
TREM_BAR_5      =   23199
DEAD_NOTE       =   23200
SLIDE_NOTE_1    =   23201
SLIDE_NOTE_2    =   23202
SLIDE_NOTE_3    =   23203
SLIDE_NOTE_4    =   23204
SLIDE_NOTE_5    =   23205
SLIDE_NOTE_6    =   23206
HAMMER          =   23207
VIBRATO         =   23208
HARMONIC_1      =   23209

#################################################################################
MEASURE         =   8
VERBOSE         =   0
