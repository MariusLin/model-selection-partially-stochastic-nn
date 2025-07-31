__all__ = ['rbf', 'linear', 'sin', 'cos', 'swish']

import torch
import torch.nn.functional as F
import math

"""
This is a slight adaptation from Tran et al. 2022
Define activation functions for neural network
"""
# RBF function
rbf = lambda x: torch.exp(-x**2)

# Linear function
linear = lambda x: x

# Sin function
sin = lambda x: torch.sin(x)

# Cos function
cos = lambda x: torch.cos(x)

# Swiss function
swish = lambda x: x * torch.sigmoid(x)

softplus = lambda x: F.softplus(x)
# This is the normal softplus function moved to be 0 at 0 and squared to always be positive
adapted_softplus = lambda x: (F.softplus(x) - F.softplus(torch.zeros_like(x)))**2
