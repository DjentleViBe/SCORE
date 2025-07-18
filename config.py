"""Defines parameters required for training and inference"""
TUNING              =   [53, 48, 44, 39, 34, 27]
################################ transformers #################################
# 0: train, 
# 1 : eval
# 2 : both 
# 3 : load from file
MODE                =   2
BACKUP              =   "maestro_gpro-v1.0.0_Large"
SAVE                =   "maestro_gpro-v1.0.0_Large_L"
BOS_TRUE            =   0
EOS_TRUE            =   1
# 0 - Vanilla
# 1 - Disregard any note > BARRE_NOTE
# 2 - n bars of generation, n = TEST_TRIES
# 3 - Same as above + last note of every bar is fed to firt note of next bar
# 4 - Same as above + loop until no repeatitive notes > BOS 
TEST_CRITERIA       =   3
TEST_TRIES          =   10
# Total number of start_id to try with testing
TEST_NUM            =   3
# 1 - Greedy Search
# 2 - Beam Search
# 3 - Multinomial Sampling
# 4 - Multinomial Sampling - manual
PREDICTION_CRITERIA =   3
########## Params ##############
EPOCHS          =   100
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
TRAINING        =   ["maestro_gpro-v1.0.0"]
NUM_SEQUENCE    =   6208
CONVERGENCE     =   0.0005
TEMPERATURE     =   1.0
BATCH           =   32
OVERLAP         =   8
LAMBDA          =   1.5
ALPHA           =   0.1
DELTA           =   0.001

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
