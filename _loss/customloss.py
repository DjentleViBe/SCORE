#pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments, too-many-branches
"""
Custom loss functions for sequence modeling with repetition penalties.
"""
import sys
from collections import Counter
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import config as cfg
from models import pairwise_l2
np.set_printoptions(threshold=sys.maxsize)

class SequenceMemory(nn.Module):
    """
    Sequence Memory Module to maintain embeddings of sequences.
    """
    def __init__(self, NUM_SEQUENCE):
        super().__init__()
        embedding_dim = cfg.D_MODEL
        self.seq_embed = nn.Embedding(NUM_SEQUENCE, embedding_dim)

    def forward(self, seq_ids):
        """Forward pass"""
        return self.seq_embed(seq_ids)

def generate_sampled_sequence(decoder, inputs, embedding_layer,
                              pos_enc, seq_len, temperature=1.0):
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
    """Repetition Penalty Loss for Specific Tokens 
    with Sequence Embedding Regularization"""
    def __init__(self, label_smoothing=0.0, repetition_penalty_weight=1.0,
                 ngram_size=1, penalize_tokens=None, sequence_penalty_weight=1.0):
        super().__init__()
        # Assuming 0 is pad_token_id
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=0)
        self.repetition_penalty_weight = repetition_penalty_weight
        self.sequence_penalty_weight = sequence_penalty_weight
        self.ngram_size = ngram_size
        self.penalize_tokens = penalize_tokens if penalize_tokens else []

    def forward(self, logits, targets, prev_sequence_embedding, seq_mem):
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
        """
        logits, hidden_states, _ = logits  # unpack the tuple

        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        ce_loss = self.ce_loss(logits_flat, targets_flat)

        hidden_tensor = hidden_states[0]
        #print(hidden_tensor.shape) # BATCH, H, SEQ_LEN, HIDDEN
        #print(seq_mem.shape) # BATCH, SEQ_LEN
        b_val, h_val, t_val, d = hidden_tensor.shape
        hidden_tensor = hidden_tensor.permute(0, 2, 1, 3).reshape(b_val, t_val, h_val * d)
        sequence_embedding = hidden_tensor.mean(dim=1) + seq_mem  # (B, seq_len*H1*H2)

        if prev_sequence_embedding is not None and sequence_embedding.shape[0] == cfg.BATCH:
            d_curr = pairwise_l2(sequence_embedding)   # distances
            d_prev = pairwise_l2(prev_sequence_embedding)
            rel_loss = F.mse_loss(d_curr, d_prev.detach()) * self.sequence_penalty_weight
        else:
            rel_loss = torch.tensor(0.0, device=sequence_embedding.device)
        repetition_loss = self.repetition_penalty_weight * \
                        soft_repetition_penalty(logits, temperature=cfg.TEMPERATURE)
        total_loss = ce_loss + repetition_loss +  rel_loss
        return total_loss, ce_loss, repetition_loss, rel_loss, sequence_embedding.detach()

    def compute_repetition_penalty(self, generated_tokens):
        """
        Calculates repetition penalty based on repeated n-grams for specific tokens,
        normalized by total number of n-grams.
        """
        batch_penalty = 0.0
        total_ngrams_count = 0  # Total ngrams in batch
        for seq in generated_tokens:
            seq_list = seq.tolist()
            ngrams = [tuple(seq_list[i:i+self.ngram_size]) for
                      i in range(len(seq_list) - self.ngram_size + 1)]
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

class ValidationLoss(nn.Module):
    """Validation Loss using standard Cross Entropy"""
    def forward(self, logits, targets):
        """Forward pass"""
        return F.cross_entropy(logits, targets, ignore_index=0)
