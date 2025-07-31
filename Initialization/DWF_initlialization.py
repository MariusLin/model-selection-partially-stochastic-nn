import torch

"""
It is an implementation of algorithm D.1 from Kolb et al. 2025
"""

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
    omega = []
    for l, (n_l, sigma_w) in enumerate(zip(layer_sizes, sigma_w_list)):
        sigma_l = sigma_w ** (1.0 / D)
        omega_min = epsilon ** (1.0 / D)
        omega_max = min(1.0, (2 * sigma_w) ** (1.0 / D))

        # Initialize empty tensor
        omega_l = torch.empty((n_l, D), device=device)

        for d in range(D):
            count = 0
            # Rejection sampling for each column
            while count < n_l:
                # Generate a batch of candidate samples
                batch_size = 2 * (n_l - count)  # over-generate to improve efficiency
                samples = torch.randn(batch_size, device=device) * sigma_l
                mask = (torch.abs(samples) > omega_min) & (torch.abs(samples) < omega_max)
                valid_samples = samples[mask]

                # Fill in as many valid samples as possible
                num_valid = valid_samples.size(0)
                num_to_fill = min(num_valid, n_l - count)
                if num_to_fill > 0:
                    omega_l[count:count+num_to_fill, d] = valid_samples[:num_to_fill]
                    count += num_to_fill

        omega.append(omega_l)

    return omega
