import torch
import config as cfg

def create_combined_mask(input_ids, device, seq_length, num_heads):
    batch_size = input_ids.size(0)

    # 1. Causal mask: upper triangle with -inf, shape (1, 1, seq_len, seq_len)
    causal_mask = torch.triu(
        torch.full((seq_length, seq_length), float('-inf'), device=device),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    # 2. Padding mask: mask future tokens for padded positions
    # Shape: (batch_size, 1, 1, seq_len)
    pad_mask = (input_ids == 0).unsqueeze(1).unsqueeze(2)  # bool mask
    pad_mask = pad_mask.expand(batch_size, 1, seq_length, seq_length)
    pad_mask = pad_mask.to(dtype=torch.float32) * float('-1e9')  # use -1e9 instead of -inf

    # 3. Combine masks: (batch_size, 1, seq_len, seq_len)
    combined_mask = causal_mask + pad_mask
    combined_mask = combined_mask.expand(batch_size, num_heads, seq_length, seq_length)

    return combined_mask


