import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from Initialization.DWF_initlialization import dwf_initialization


class FactorizedLinear(nn.Module):
    def __init__(self, n_in, n_out, D, W_std = None, b_std = None, scaled_variance=True, device = "cpu"):
        """Initialization.

        Args:
            n_in: int, the size of the input data.
            n_out: int, the size of the output.
        """
        super(FactorizedLinear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.scaled_variance = scaled_variance
        self.D = D
        self.device = device

        if W_std is None:
            if self.scaled_variance:
                W_std = 1.
            else:
                W_std = 1. / math.sqrt(self.n_in)
        if b_std is None:
            b_std = 1.

        # Initialize the parameters
        self.W_omega = nn.ParameterList()
        self.b_omega = nn.ParameterList()


        omega_factors_W_std = dwf_initialization(
            [self.n_in*self.n_out], [W_std], self.D, W_std*0.25, device = self.device
        )[0]  # shape (layer_size, D)
        # Store D separate factor tensors
        for d in range(self.D):
            param = nn.Parameter(omega_factors_W_std[:, d].view((n_in, n_out)), requires_grad=True) 
            self.W_omega.append(param)
        
        omega_factors_b_std = dwf_initialization(
            [n_out], [b_std], self.D, b_std*0.25, device = self.device
        )[0]  # shape (layer_size, D)
        # Store D separate factor tensors
        for d in range(self.D):
            param = nn.Parameter(omega_factors_b_std[:, d].view(n_out), requires_grad=True) 
            self.b_omega.append(param)

    def forward(self, X):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.

        Returns:
            output: torch.tensor, [batch_size, output_dim], the output data.
        """
        X = X.float()
        W = self.get_W()
        if self.scaled_variance:
            W = W / math.sqrt(self.n_in)
        b = self.get_b()
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
        Ws = self.get_W().repeat(n_samples, 1, 1)
        bs = self.get_b().repeat(n_samples, 1, 1)

        return torch.matmul(X, Ws) + bs
    
    def get_W(self, eps_tiny = 1.19e-7):
        """
        Multiplies all the weight matrices elementwise
        """
        W = self.W_omega[0]
        for d in range(1, self.D):
            W = W * self.W_omega[d]
        W[torch.abs(W) < eps_tiny] = 0.0
        return W

    def get_b(self, eps_tiny = 1.19e-7):
        """
        Multiplies all the bias vectors elementwise
        """
        b = self.b_omega[0]
        for d in range(1, self.D):
            b = b * self.b_omega[d]
        b[torch.abs(b) < eps_tiny] = 0.0
        return b
    
    def get_num_pruned_W(self):
        """
        Returns the number of pruned weights
        """
        W = self.get_W()
        return torch.sum(W == 0).item()
    
    def get_num_pruned_b(self):
        """
        Returns the number of pruned biases
        """
        b = self.get_b()
        return torch.sum(b == 0).item()