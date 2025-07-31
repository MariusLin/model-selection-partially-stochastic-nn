import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

"""
This is a slight adaptation of the linear layer taken from Tran et al. 2022
Here, the parameters that are to be treated stochastically are getting the suffix stoch
"""
class StochLinear(nn.Module):
    def __init__(self, n_in, n_out, scaled_variance=True):
        """Initialization.

        Args:
            n_in: int, the size of the input data.
            n_out: int, the size of the output.
        """
        super(StochLinear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.scaled_variance = scaled_variance

        # Initialize the parameters
        self.W_stoch = nn.Parameter(torch.zeros(self.n_in, self.n_out), True)
        self.b_stoch = nn.Parameter(torch.zeros(self.n_out), True)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.
        if not self.scaled_variance:
            std = std / math.sqrt(self.n_in)
        init.normal_(self.W_stoch, 0, std)
        init.constant_(self.b_stoch, 0)

    def forward(self, X):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.

        Returns:
            output: torch.tensor, [batch_size, output_dim], the output data.
        """
        W = self.W_stoch
        if self.scaled_variance:
            W = W / math.sqrt(self.n_in)
        b = self.b_stoch
        return torch.mm(X, W) + b
    
    def sample_predict(self, X, n_samples):
        """Makes predictions using a set of sampled weights.

        Args:
            X: torch.tensor, [n_samples, batch_size, input_dim], the input
                data.
            n_samples: int, the number of weight samples used to make
                predictions.

        Returns:
            torch.tensor, [n_samples, batch_size, output_dim], the output data.
        """
        X = X.float()
        Ws = self.W_stoch.repeat(n_samples, 1, 1)
        bs = self.b_stoch.repeat(n_samples, 1, 1)
        return torch.matmul(X, Ws) + bs