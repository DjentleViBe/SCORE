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

def generate_sampled_sequence(decoder, input_seq, embedding_layer, pos_enc, mask, seq_len, temperature=1.0):
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
    batch_size = input_seq.size(0)
    sampled_seq = input_seq.clone()

    for t in range(seq_len, seq_len):
        embeddings = embedding_layer(sampled_seq)
        output = decoder(embeddings + pos_enc, mask)
        logits = output[:, t-1, :]  # logits for the last generated token
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        
        next_token = multinomial_sample_2(probs, num_samples = 40)
        sampled_seq[:, t] = next_token

    return sampled_seq

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
        batch_size, seq_len, _ = logits.size()
        logits_flat = logits.view(-1, cfg.VOCAB_SIZE)
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
