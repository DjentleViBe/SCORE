#pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments, too-many-branches
"""Run inference until criterion is reached"""
import numpy as np
import torch
import torch.nn.functional as F
from masking import create_combined_mask
from config import EOS, BAR

import config as cfg
from postprocess import decoder_inference

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
    elif cfg.TEST_CRITERIA in (3, 4):
        dummy_out = np.zeros((cfg.TEST_TRIES, cfg.MAX_SEQ_LENGTH), dtype = 'int32')
        t = 0
        while t < cfg.TEST_TRIES:
            print(f"Testing -> {t}")
            dummy_out[t] = decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask,
                                     cfg.MAX_SEQ_LENGTH, device).cpu().numpy()
            if np.any(dummy_out[t] == 0):
                print("0 detected")
                break
            idx = len(dummy_out[t]) - 1
            while idx >= 0 and dummy_out[t][idx] >= BAR:
                idx -= 1
            if idx >= 0:
                dummy_in[0][0] = dummy_out[t][idx]
            else:
                dummy_in[0][0] = EOS
            #dummy_out[t] = [15485, 27051, 15600, 11600, 11600, 11618, 15527,
            # 27051, 11649, 27051,
            #7846, 27051, 7798, 11642, 11755, 11827]
            t += 1

    return dummy_out

def testing(loader_test, device, embedding_layer, decoder, pos_enc):
    """
    Run testing epoch
    
    :param loader_test: test data loader
    :param device: device name
    :param embedding_layer: embedding layer
    :param decoder: decoder model
    :param pos_enc: position encoding
    """
    decoder.eval()
    with torch.no_grad():
        epoch_loss = 0
        batch_count = 0
        for batch_token_ids, batch_target in loader_test:
            batch_token_ids = batch_token_ids.to(device)
            batch_target = batch_target.to(device)

            mask = create_combined_mask(batch_token_ids, device,
                                        seq_length=cfg.MAX_SEQ_LENGTH, num_heads=cfg.NUM_HEADS)
            embeddings = embedding_layer(batch_token_ids)
            input_embeddings = embeddings + pos_enc[:batch_token_ids.size(1)]

            logits, *_ = decoder(input_embeddings, mask)
            logits = logits.view(-1, logits.size(-1))
            targets = batch_target.view(-1)

            # Cross-entropy per token
            loss = F.cross_entropy(logits, targets)
            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
    return avg_loss
