"""
Ploting training data distributions
"""
from postprocess import plotbar, plotbarlog, readbincount
import config as cfg

bincounts_train = readbincount('./RESULTS/' + cfg.BACKUP + "/"
                               + cfg.BACKUP + '_trainingprobability.txt')
bincountsbeats_train = readbincount('./RESULTS/' + cfg.BACKUP + "/"
                                    + cfg.BACKUP + '_trainingbeatprobability.txt')
bincountsbeattype_train = readbincount('./RESULTS/' + cfg.BACKUP + "/"
                                       + cfg.BACKUP + '_trainingbeattypeprobability.txt')
bincountsaccents_train = readbincount('./RESULTS/' + cfg.BACKUP + "/"
                                      + cfg.BACKUP + '_trainingaccentprobability.txt')

plotbar(cfg.labelsnotes, 'Occurance of Notes', bincounts_train,
        './RESULTS/' + cfg.BACKUP + "/" + cfg.BACKUP + '_trainingprobability.pdf')
plotbar(cfg.labelsbeats, 'Occurance of Beats', bincountsbeats_train,
        './RESULTS/' + cfg.BACKUP + "/" + cfg.BACKUP + '_trainingbeatprobability.pdf')
plotbar(cfg.labelsbeattype, 'Occurance of Beat Type', bincountsbeattype_train,
        './RESULTS/' + cfg.BACKUP + "/" + cfg.BACKUP + '_trainingbeattypeprobability.pdf')
plotbarlog(cfg.labelsaccents, 'Occurance of Accents', bincountsaccents_train,
           './RESULTS/' + cfg.BACKUP + "/" + cfg.BACKUP + '_trainingaccentprobability.pdf')
