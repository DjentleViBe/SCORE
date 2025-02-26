import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg

class RepetitionPenaltyLossForSpecificTokens(nn.Module):
    def __init__(self, label_smoothing=0.0, repetition_penalty_weight=1.0, ngram_size=3, penalize_tokens=None):
        super(RepetitionPenaltyLossForSpecificTokens, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=0)  # Assuming 0 is pad_token_id
        self.repetition_penalty_weight = repetition_penalty_weight
        self.ngram_size = ngram_size
        self.penalize_tokens = penalize_tokens if penalize_tokens else []

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
        """
        batch_size, seq_len, _ = logits.size()
        
        targets_flat = targets.view(-1)

        with torch.no_grad():
            generated_tokens = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)

        # Compute repetition penalty on logits before softmax
        repetition_penalty = self.compute_repetition_penalty(logits)  # Now using logits directly
        logits = logits - self.repetition_penalty_weight * repetition_penalty  # Apply penalty BEFORE softmax

        logits_flat = logits.view(-1, cfg.VOCAB_SIZE)
        ce_loss = self.ce_loss(logits_flat, targets_flat)
        # Compute repetition loss as a separate term
        repetition_loss = repetition_penalty.mean()  # Take mean over batch


        total_loss = ce_loss + self.repetition_penalty_weight * repetition_loss
        return total_loss, ce_loss, repetition_loss

    def compute_repetition_penalty(self, generated_tokens):
        """
        Calculates repetition penalty based on repeated n-grams for specific tokens.
        """
        batch_penalty = 0.0
        for seq in generated_tokens:
            seq_list = seq.tolist()

            # Only penalize for tokens that are in `self.penalize_tokens`
            penalized_tokens = [token for token in seq_list if token in self.penalize_tokens]
            
            # If we have tokens to penalize, apply ngram check
            if penalized_tokens:
                ngrams = [tuple(seq_list[i:i+self.ngram_size]) for i in range(len(seq_list)-self.ngram_size+1)]
                ngram_counts = {}

                for ngram in ngrams:
                    ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

                # Penalize repeated n-grams only for penalized tokens
                repetition_count = sum(count - 1 for count in ngram_counts.values() if count > 1)
                batch_penalty += repetition_count

        # Normalize by batch size
        return torch.tensor(batch_penalty / generated_tokens.size(0), requires_grad=True).to(generated_tokens.device)
