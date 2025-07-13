"""Run inference until criterion is reached"""
import config as cfg
from postprocess import decoder_inference
import torch
import numpy as np
from config import EOS, BAR

def inference(start, device, decoder, embedding_layer, pos_enc, mask):
    """Run inference"""
    dummy_np = np.full((1, cfg.MAX_SEQ_LENGTH), cfg.EOS, dtype = 'int32')
    if cfg.BOS_TRUE == 0:
        dummy_np[0, 0] = start
    else:
        dummy_np[0, 0] = cfg.BOS
        dummy_np[0, 1] = start
    dummy_in = torch.tensor(dummy_np).to(device)

    if cfg.TEST_CRITERIA == 0:
        dummy_out = decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask,
                                     cfg.MAX_SEQ_LENGTH, device).cpu().numpy()
    elif cfg.TEST_CRITERIA == 1:
        for t in range (0, cfg.TEST_TRIES):
            print(f"Testing -> {t}")
            
            dummy_out = decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask,
                                     cfg.MAX_SEQ_LENGTH, device).cpu().numpy()
            #dummy_out[0][4] = 26416
            if np.any(dummy_out.cpu().numpy() > cfg.BARRE_NOTE):
                return dummy_out.cpu().numpy()
    elif cfg.TEST_CRITERIA == 2:
        dummy_out = np.zeros((cfg.TEST_TRIES, cfg.MAX_SEQ_LENGTH), dtype = 'int32')
        t = 0
        while t < cfg.TEST_TRIES:
            print(f"Testing -> {t}")
            dummy_out[t] = decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask,
                                     cfg.MAX_SEQ_LENGTH, device).cpu().numpy()
            if np.any(dummy_out[t] == 0):
                print("0 detected")
            else:
                t += 1
    elif cfg.TEST_CRITERIA == 3 or cfg.TEST_CRITERIA == 4:
        dummy_out = np.zeros((cfg.TEST_TRIES, cfg.MAX_SEQ_LENGTH), dtype = 'int32')
        t = 0
        while t < cfg.TEST_TRIES:
            print(f"Testing -> {t}")
            dummy_out[t] = decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask,
                                     cfg.MAX_SEQ_LENGTH, device).cpu().numpy()
            if np.any(dummy_out[t] == 0):
                print("0 detected")
                break
            else:
                idx = len(dummy_out[t]) - 1
                while idx >= 0 and dummy_out[t][idx] >= BAR:
                    idx -= 1
                if idx >= 0:
                    dummy_in[0][0] = dummy_out[t][idx]
                else:
                    dummy_in[0][0] = EOS
                #dummy_out[t] = [13618, 27049, 21301,  7959, 11687, 27051, 25219, 13595, 11846, 15574,
                #23262, 15552,  8514, 12378, 11875, 23207, 11688, 11688, 13615, 13638,
                #15780, 22195, 13758, 13781, 11617, 11687,  7979, 15689, 15780, 13618,
                #13618, 13618]
                t += 1
                
   
    return dummy_out
