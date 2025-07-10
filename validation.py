import torch
import config as cfg
from masking import create_combined_mask

def validation(loader, device, optimizer, embedding_layer, decoder, criterion, pos_enc):
    decoder.eval()
    epoch_loss = 0.0
    batch_count = 0
    prev_sequence_embedding = None
    with torch.no_grad():
        for batch_token_ids, batch_target in loader:
            #if len(batch_token_ids) != cfg.BATCH:
            #    break
            batch_token_ids = batch_token_ids.to(device)
            batch_target = batch_target.to(device)
            mask = create_combined_mask(batch_token_ids, device, seq_length=cfg.MAX_SEQ_LENGTH, num_heads=cfg.NUM_HEADS)
                
            optimizer.zero_grad()
            # Embeddings
            embeddings = embedding_layer(batch_token_ids)
            input_embeddings = embeddings + pos_enc[:batch_token_ids.size(1)]

            # Forward
            logits = decoder(input_embeddings, mask)

            # Flatten target
            batch_target = batch_target.view(-1)

            # Compute loss
            loss, ce_loss, rep_loss, kl_loss, prev_sequence_embedding = criterion(
                logits, batch_target, prev_sequence_embedding)
            epoch_loss += loss.item()
            batch_count += 1
        avg_loss = epoch_loss / batch_count
    return avg_loss