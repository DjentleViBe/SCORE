"""Run inference until criterion is reached"""
import config as cfg
from postprocess import decoder_inference
import torch
import numpy as np
from config import BOS

def inference(device, decoder, embedding_layer, pos_enc, mask):
    """Run inference"""
    dummy_np = np.full((1, cfg.MAX_SEQ_LENGTH), cfg.EOS, dtype = 'int32')
    if cfg.BOS_TRUE == 0:
        dummy_np[0, 0] = cfg.START_ID
    else:
        dummy_np[0, 0] = cfg.BOS
        dummy_np[0, 1] = cfg.START_ID
    dummy_in = torch.tensor(dummy_np).to(device)

    if cfg.TEST_CRITERIA == 0:
        dummy_out = decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask,
                                     cfg.MAX_SEQ_LENGTH).cpu().numpy()
    elif cfg.TEST_CRITERIA == 1:
        for t in range (0, cfg.TEST_TRIES):
            print(f"Testing -> {t}")
            
            dummy_out = decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask,
                                     cfg.MAX_SEQ_LENGTH).cpu().numpy()
            #dummy_out[0][4] = 26416
            if np.any(dummy_out.cpu().numpy() > cfg.BARRE_NOTE):
                return dummy_out.cpu().numpy()
    elif cfg.TEST_CRITERIA == 2:
        dummy_out = np.zeros((cfg.TEST_TRIES, cfg.MAX_SEQ_LENGTH), dtype = 'int32')
        t = 0
        while t < cfg.TEST_TRIES:
            print(f"Testing -> {t}")
            dummy_out[t] = decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask,
                                     cfg.MAX_SEQ_LENGTH).cpu().numpy()
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
                                     cfg.MAX_SEQ_LENGTH).cpu().numpy()
            if np.any(dummy_out[t] == 0):
                print("0 detected")
            else:
                if dummy_out[t][-1] < BOS:
                    dummy_in[0][0] = dummy_out[t][-1]
                t += 1

    return dummy_out
