#pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments, too-many-branches
"""
Validation loop
"""
import torch
import config as cfg
from masking import create_combined_mask
from _loss.customloss import ValidationLoss

def validation(loader, device, optimizer, embedding_layer, decoder, pos_enc):
    """
    Run validation loop
    
    :param loader: validation data loader
    :param device: device name
    :param optimizer: optimizer
    :param embedding_layer: embedding layer
    :param decoder: decoder model
    :param criterion: loss function
    :param pos_enc: position encoding
    """
    decoder.eval()
    epoch_loss = 0.0
    batch_count = 0
    validation_criterion = ValidationLoss()
    with torch.no_grad():
        for batch_token_ids, batch_target in loader:
            #if len(batch_token_ids) != cfg.BATCH:
            #    break
            batch_token_ids = batch_token_ids.to(device)
            batch_target = batch_target.to(device)
            mask = create_combined_mask(batch_token_ids, device,
                                        seq_length=cfg.MAX_SEQ_LENGTH, num_heads=cfg.NUM_HEADS)

            optimizer.zero_grad()
            # Embeddings
            embeddings = embedding_layer(batch_token_ids)
            input_embeddings = embeddings + pos_enc[:batch_token_ids.size(1)]

            # Forward
            logits, *_ = decoder(input_embeddings, mask)
            logits = logits.view(-1, logits.size(-1))
            # Flatten target
            batch_target = batch_target.view(-1)

            # Compute loss
            loss = validation_criterion(logits, batch_target)
            epoch_loss += loss.item()
            batch_count += 1
        avg_loss = epoch_loss / batch_count
    return avg_loss
