"""
Implements the algorithm D.1 from Kolb et al. 2025
"""

import torch

def dwf_initialization(layer_sizes, sigma_w_list, D, epsilon, device='cpu'):
    """
    Parameters:
    - layer_sizes: List[int], number of weights (n_l) per layer
    - sigma_w_list: List[float], standard deviations σ_{w,l} for each layer
    - D: int, factorization depth
    - epsilon: float, minimum absolute value ε
    - device: str, 'cpu' or 'cuda'

    Returns:
    - omega: List[torch.Tensor], each is (n_l, D) tensor for layer l
    """
    L = len(layer_sizes)
    omega = []
    for l in range(L):
        n_l = layer_sizes[l]
        sigma_w = sigma_w_list[l]
        sigma_l = sigma_w ** (1.0 / D)
        omega_min = epsilon ** (1.0 / D)
        omega_max = min(1.0, (2 * sigma_w) ** (1.0 / D))

        # Prepare tensor to hold factors: shape (n_l, D)
        omega_l = torch.empty((n_l, D), device=device)

        for j in range(n_l):
            for d in range(D):
                # Rejection sampling for valid omega
                while True:
                    sample = torch.randn(1, device=device) * sigma_l
                    if omega_min < torch.abs(sample) < omega_max:
                        omega_l[j, d] = sample
                        break

        omega.append(omega_l)

    return omega