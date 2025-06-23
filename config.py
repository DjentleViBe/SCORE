"""Defines parameters required for training and inference"""

################################ transformers #################################
# 0: train, 
# 1 : eval
# 2 : both 
# 3 : load from file
MODE                =   0
BACKUP              =   "dec_only_notes_48"
SAVE                =   "dec_only_notes_48A"
START_ID            =   9437
BOS_TRUE            =   1
EOS_TRUE            =   1
# 0 - Vanilla
# 1 - Disregard any note > BARRE_NOTE
# 2 - n bars of generation, n = TEST_TRIES
# 3 - Same as above + last note of every bar is fed to firt note of next bar
# 4 - Same as above + loop until no repeatitive notes > BOS 
TEST_CRITERIA       =   3
TEST_TRIES          =   10
# 1 - Greedy Search
# 2 - Beam Search
# 3 - Multinomial Sampling
# 4 - Multinomial Sampling - manual
PREDICTION_CRITERIA =   3
########## Params ##############
EPOCHS          =   200
SAVE_EVERY      =   100
VOCAB_SIZE      =   26430
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
NUM_SEQUENCE    =   6208
CONVERGENCE     =   0.0005
TEMPERATURE     =   1.0
BATCH           =   32

EOS             =   26405
BOS             =   26406
BARRE_NOTE      =   26407
BEND_NOTE_1     =   26408
BEND_NOTE_2     =   26409
BEND_NOTE_3     =   26410
BEND_NOTE_4     =   26411
BEND_NOTE_5     =   26412
BEND_NOTE_6     =   26413
BEND_NOTE_7     =   26414
TREM_BAR_1      =   26415
TREM_BAR_2      =   26416
TREM_BAR_3      =   26417
TREM_BAR_4      =   26418
TREM_BAR_5      =   26419
DEAD_NOTE       =   26420
SLIDE_NOTE_1    =   26421
SLIDE_NOTE_2    =   26422
SLIDE_NOTE_3    =   26423
SLIDE_NOTE_4    =   26424
SLIDE_NOTE_5    =   26425
SLIDE_NOTE_6    =   26426
HAMMER          =   26427
VIBRATO         =   26428
HARMONIC_1      =   26429

#################################################################################
MEASURE         =   8
