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

    for step in range(inputs.size(1), seq_len):
        # Embed the last generated token(s)
        input_embeds = embedding_layer(generated_tokens[:, step-1:step])  # (batch, 1, d_model)
        # Add positional encoding if needed (adjust shape accordingly)
        if pos_enc is not None:
            pos_embed = pos_enc(step - 1).unsqueeze(0).expand(batch_size, 1, -1)  # (batch, 1, d_model)
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

class RepetitionPenaltyLossForSpecificTokens(nn.Module):
    def __init__(self, label_smoothing=0.0, repetition_penalty_weight=1.0, ngram_size=1, penalize_tokens=None):
        super(RepetitionPenaltyLossForSpecificTokens, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=0)  # Assuming 0 is pad_token_id
        self.repetition_penalty_weight = repetition_penalty_weight
        self.ngram_size = ngram_size
        self.penalize_tokens = penalize_tokens if penalize_tokens else []

    def forward(self, logits, targets, decoder=None, inputs=None, embedding_layer=None, pos_enc=None, mask=None):
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
        """
        # batch_size, seq_len, _ = logits.size()
        logits, _, _ = logits  # unpack the tuple
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        ce_loss = self.ce_loss(logits_flat, targets_flat)

        with torch.no_grad():
            sampled_tokens = generate_sampled_sequence(decoder, inputs, embedding_layer, pos_enc, mask, seq_len=logits.size(1), temperature=cfg.TEMPERATURE)
        repetition_loss = self.compute_repetition_penalty(sampled_tokens)
        total_loss = ce_loss + self.repetition_penalty_weight * repetition_loss
        return total_loss, ce_loss, repetition_loss

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
