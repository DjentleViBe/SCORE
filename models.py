import torch 

def compute_sequence_gaussians(embedded_sequences, eps=1e-8):
    """
    embedded_sequences: Tensor of shape [N, T, d]
        - N: number of sequences
        - T: tokens per sequence
        - d: embedding dimension
    Returns:
        mu_all: [N, d] (mean per sequence)
        sigma_all: [N, d] (stddev per sequence)
    """
    mu_all = embedded_sequences.mean(dim=1)  # Mean over token axis → shape: [N, d]
    var_all = embedded_sequences.var(dim=1, unbiased=False)  # Variance over tokens → [N, d]
    sigma_all = torch.sqrt(var_all + eps)  # Std dev for stability
    return mu_all, sigma_all

def kl_divergence_gaussians(mu_q, sigma_q, mu_k, sigma_k, eps=1e-8):
    """
    mu_q, sigma_q: [N, d]
    mu_k, sigma_k: [M, d]
    
    Returns:
        kl_matrix: [N, M] with KL divergence between sequence q_i and k_j
    """
    N, d = mu_q.shape
    M = mu_k.shape[0]

    mu_q = mu_q.unsqueeze(1)         # [N, 1, d]
    sigma_q = sigma_q.unsqueeze(1)   # [N, 1, d]

    mu_k = mu_k.unsqueeze(0)         # [1, M, d]
    sigma_k = sigma_k.unsqueeze(0)   # [1, M, d]

    var_q = sigma_q ** 2 + eps       # [N, 1, d]
    var_k = sigma_k ** 2 + eps       # [1, M, d]

    log_term = torch.log(var_k / var_q)            # [N, M, d]
    frac_term = var_q / var_k                       # [N, M, d]
    mean_term = ((mu_q - mu_k) ** 2) / var_k       # [N, M, d]

    kl = 0.5 * torch.sum(log_term + frac_term + mean_term - 1, dim=2)  # [N, M]

    return kl

def kl_divergence_gaussians_loss(mu_q, sigma_q, mu_k, sigma_k, eps=1e-8):
    """
    mu_q, sigma_q: [B, d]
    mu_k, sigma_k: [B, d]
    Returns:
        kl: [B]
    """
    var_q = sigma_q ** 2 + eps
    var_k = sigma_k ** 2 + eps

    log_term = torch.log(var_k / var_q)
    frac_term = var_q / var_k
    mean_term = ((mu_q - mu_k) ** 2) / var_k

    kl = 0.5 * torch.mean(log_term + frac_term + mean_term - 1, dim=-1)  # sum over d, result [B]
    return kl