"""various inference methods"""
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import config as cfg

def multinomial_sample(probabilities, num_samples):
    """Multinomial sampling"""
    # Calculate the cumulative sum of probabilities
    cumulative_probabilities = []
    cumulative_sum = 0.0
    for p in probabilities[0]:
        # print(p.item())
        cumulative_sum += p.item()
        cumulative_probabilities.append(cumulative_sum)

    #plt.plot(cumulative_probabilities, color = 'k')
    #plt.suptitle("Cumulative probability")
    #plt.xlabel("Tokens")
    #plt.ylabel("Probability")
    # plt.show()

    samples = []

    for _ in range(num_samples):
        # Generate a random number between 0 and 1
        r = random.random()
        # Find the interval that r falls into
        for i, cumulative_probability in enumerate(cumulative_probabilities):
            if r < cumulative_probability:
                samples.append(i)
                break
    
    # Choose one sample from the list of samples
    chosen_sample = random.choice(samples)
    return chosen_sample


def multinomial_sample_2(probs, num_samples):
    return torch.multinomial(probs, num_samples=num_samples, replacement=True)[0]


def create_causal_mask(seq_len, device):
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    mask.unsqueeze(0).unsqueeze(0) 
    mask = mask.expand(1, cfg.NUM_HEADS, seq_len, seq_len) 
    return mask