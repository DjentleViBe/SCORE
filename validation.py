import torch
import config as cfg

def validation(loader, device, optimizer, embedding_layer, decoder, mask, criterion, pos_enc):
    decoder.eval()

    epoch_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch_token_ids, batch_target in loader:
            batch_token_ids = batch_token_ids.to(device)
            batch_target = batch_target.to(device)
            # Prepare `inputs` tensor (e.g., for teacher forcing or decoder input)
            inputs = torch.zeros((batch_token_ids.size(0), cfg.MAX_SEQ_LENGTH), dtype=torch.long).to(device)
            inputs[:, 0] = cfg.START_ID

            optimizer.zero_grad()
            # Embeddings
            embeddings = embedding_layer(batch_token_ids)
            input_embeddings = embeddings + pos_enc[:batch_token_ids.size(1)]

            # Forward
            logits = decoder(input_embeddings, mask)

            # Flatten target
            batch_target = batch_target.view(-1)

            # Compute loss
            loss, ce_loss, rep_loss = criterion(
                logits, batch_target, decoder, inputs, embedding_layer, pos_enc
            )
            epoch_loss += loss.item()
            batch_count += 1
        avg_loss = epoch_loss / batch_count
    return avg_loss