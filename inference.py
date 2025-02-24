"""various inference methods"""
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def multinomial_sample(probabilities, num_samples):
    """Multinomial sampling"""
    # Calculate the cumulative sum of probabilities
    cumulative_probabilities = []
    cumulative_sum = 0.0
    for p in probabilities[0]:
        # print(p.item())
        cumulative_sum += p.item()
        cumulative_probabilities.append(cumulative_sum)

    plt.plot(cumulative_probabilities, color = 'k')
    plt.suptitle("Cumulative probability")
    plt.xlabel("Tokens")
    plt.ylabel("Probability")
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
