import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
from collections import Counter
from postprocess import decoder_search
from inference import multinomial_sample_2
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import torch.nn.functional as F
from models import compute_sequence_gaussians, kl_divergence_gaussians_loss

def generate_sampled_sequence(decoder, inputs, embedding_layer, pos_enc, mask, seq_len, temperature=1.0):
    """
    Generate a sequence by sampling tokens multinomially from the decoder output at each step.
    Args:
        decoder: your decoder model
        input_seq: starting tensor (batch_size, seq_len) with initial tokens (e.g., start tokens)
        embedding_layer, pos_enc, mask: model components for forward pass
        seq_len: total length of sequence to generate
    Returns:
        sampled_seq: (batch_size, seq_len) tensor of sampled token indices
    """
    batch_size = inputs.size(0)
    device = inputs.device
    
    generated_tokens = inputs.clone()
    
    # Initialize KV caches as lists of None for each layer
    key_caches = None
    value_caches = None

    for step in range(1, seq_len):
        # Embed the last generated token(s)
        input_embeds = embedding_layer(generated_tokens[:, step-1:step])  # (batch, 1, d_model)
        # Add positional encoding if needed (adjust shape accordingly)
        if pos_enc is not None:
            pos_embed = pos_enc[step - 1].unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1)
            input_embeds = input_embeds + pos_embed

        # Run decoder for one step with cache
        logits, key_caches, value_caches = decoder(
            input_embeds,
            decoder_mask=None,  # or appropriate mask for decoding step
            key_caches=key_caches,
            value_caches=value_caches
        )  # logits shape: (batch, 1, vocab_size)

        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)

        # Sample or greedy (here sampling)
        next_tokens = torch.multinomial(probs, num_samples=1)  # (batch, 1)

        generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)

    return generated_tokens

def soft_repetition_penalty(logits, temperature=1.0):
    """
    logits: (batch_size, seq_len, vocab_size)
    Penalize high probability on tokens that already appeared earlier in the sequence.
    """
    probs = torch.softmax(logits / temperature, dim=-1)  # (B, T, V)
    batch_size, seq_len, vocab_size = probs.size()

    # Compute cumulative sum of probs along time dimension excluding current step
    # We'll use cumsum to get sum of all previous probs at each time step:
    cumsum_probs = probs.cumsum(dim=1)  # (B, T, V)

    # For each step t, previous sum = cumsum_probs at t-1
    # Pad a zero vector at the start for indexing convenience
    zero_pad = torch.zeros((batch_size, 1, vocab_size), device=probs.device, dtype=probs.dtype)
    prev_cumsum = torch.cat([zero_pad, cumsum_probs[:, :-1, :]], dim=1)  # (B, T, V)

    # Compute average previous probs at each time step t:
    counts = torch.arange(seq_len, device=probs.device).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    counts = counts.clamp(min=1)  # avoid division by zero at t=0 (will be ignored anyway)
    avg_prev_prob = prev_cumsum / counts  # (B, T, V)

    # Dot product between current probs and avg previous probs for each time step:
    repeat_scores = (probs * avg_prev_prob).sum(dim=-1)  # (B, T)

    # Ignore the first time step since it has no previous tokens
    repeat_scores = repeat_scores[:, 1:]  # (B, T-1)

    penalty = repeat_scores.mean()

    return penalty

class RepetitionPenaltyLossForSpecificTokens(nn.Module):
    def __init__(self, label_smoothing=0.0, repetition_penalty_weight=1.0, ngram_size=1, penalize_tokens=None, sequence_penalty_weight=1.0):
        super(RepetitionPenaltyLossForSpecificTokens, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=0)  # Assuming 0 is pad_token_id
        self.repetition_penalty_weight = repetition_penalty_weight
        self.sequence_penalty_weight = sequence_penalty_weight
        self.ngram_size = ngram_size
        self.penalize_tokens = penalize_tokens if penalize_tokens else []

    def forward(self, logits, targets, prev_sequence_embedding):
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
        """
        # batch_size, seq_len, _ = logits.size()
        logits, _, _ = logits  # unpack the tuple
        probs = torch.softmax(logits, dim=-1)
        sequence_embedding = probs
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        ce_loss = self.ce_loss(logits_flat, targets_flat)
        
        if prev_sequence_embedding is not None and sequence_embedding.shape[0] == cfg.MAX_SEQ_LENGTH:
            mu_curr, sigma_curr = compute_sequence_gaussians(sequence_embedding)
            mu_prev, sigma_prev = compute_sequence_gaussians(prev_sequence_embedding.detach())
            # print(sigma_curr.max().item())
            kl_loss = kl_divergence_gaussians_loss(mu_curr, sigma_curr, mu_prev, sigma_prev)
            kl_loss = self.sequence_penalty_weight * kl_loss.mean() / cfg.BATCH
            # print(kl_loss)
            kl_loss = torch.clamp(kl_loss, max=5.0)
        else:
            kl_loss = torch.tensor(0.0, device=sequence_embedding.device)
        
        #with torch.no_grad():
        #    sampled_tokens = generate_sampled_sequence(decoder, inputs, embedding_layer, pos_enc, mask, seq_len=logits.size(1), temperature=cfg.TEMPERATURE)
        #repetition_loss = self.compute_repetition_penalty(sampled_tokens)
        repetition_loss = self.repetition_penalty_weight * soft_repetition_penalty(logits, temperature=cfg.TEMPERATURE)
        total_loss = ce_loss + repetition_loss +  kl_loss
        return total_loss, ce_loss, repetition_loss, kl_loss, sequence_embedding.detach()

    def compute_repetition_penalty(self, generated_tokens):
        """
        Calculates repetition penalty based on repeated n-grams for specific tokens,
        normalized by total number of n-grams.
        """
        batch_penalty = 0.0
        total_ngrams_count = 0  # Total ngrams in batch
        for seq in generated_tokens:
            seq_list = seq.tolist()
            ngrams = [tuple(seq_list[i:i+self.ngram_size]) for i in range(len(seq_list) - self.ngram_size + 1)]
            total_ngrams_count += len(ngrams)

            ngram_counts = Counter(ngrams)

            repetition_count = sum(
                count - 1
                for ngram, count in ngram_counts.items()
                if count > 1 and any(tok in self.penalize_tokens for tok in ngram)
            )
            batch_penalty += repetition_count

        if total_ngrams_count == 0:
            return torch.tensor(0.0, device=generated_tokens.device)

        repetition_loss = batch_penalty / total_ngrams_count
        return torch.tensor(repetition_loss, device=generated_tokens.device)
