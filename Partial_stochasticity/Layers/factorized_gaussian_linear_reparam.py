import math
import numpy as np
import torch
import torch.nn as nn
import Utilities.activation_functions as af
from Initialization.DWF_initlialization import dwf_initialization


class FactorizedGaussianLinearReparameterization(nn.Module):
    def __init__(self, n_in, n_out, D, W_std=None,
                 b_std=None, scaled_variance=True, prior_per='layer', device = "cpu"):
        """Initialization.

        Args:
            n_in: int, the size of the input data.
            n_out: int, the size of the output.
            W_std: float, the initial value of
                the standard deviation of the weights.
            b_std: float, the initial value of
                the standard deviation of the biases.
            prior_per: str, indicates whether using different prior for
                each parameter, option `parameter`, or use the share the
                prior for all parameters in the same layer, option `layer`.
        """
        super(FactorizedGaussianLinearReparameterization, self).__init__()

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

        if prior_per == "layer":
            W_shape, b_shape = (1), (1)
        elif prior_per == "parameter":
            W_shape, b_shape = (self.n_in, self.n_out), (self.n_out)
        else:
            raise ValueError("Accepted values: `parameter` or `layer`")

        self.W_mu = 0.
        self.b_mu = 0.
        self.W_omega_std = nn.ParameterList()
        self.b_omega_std = nn.ParameterList()
        # DWF Initialization for weight 
        omega_factors_W_std = dwf_initialization(
            [self.n_in * self.n_out], [W_std], self.D, W_std*0.25, device = self.device
        )[0]  # shape (layer_size, D)
        # Store D separate factor tensors
        for d in range(self.D):
            param = nn.Parameter(omega_factors_W_std[:, d].view(*W_shape), requires_grad=True) 
            self.W_omega_std.append(param)
        
        omega_factors_b_std = dwf_initialization(
            [self.n_out], [b_std], self.D, b_std*0.25, device = self.device
        )[0]  # shape (layer_size, D)
        # Store D separate factor tensors
        for d in range(self.D):
            param = nn.Parameter(omega_factors_b_std[:, d].view((b_shape, )), requires_grad=True) 
            self.b_omega_std.append(param)


    def get_W_std(self, eps_tiny = 1.19e-7):
        """
        Multiplies all the weight matrices elementwise
        """
        W_std = self.W_omega_std[0]
        for d in range(1, self.D):
            W_std = W_std * self.W_omega_std[d]
        W_std[torch.abs(W_std) < eps_tiny] = 0.0
        return W_std

    def get_b_std(self, eps_tiny = 1.19e-7):
        """
        Multiplies all the bias vectors elementwise
        """
        b_std = self.b_omega_std[0]
        for d in range(1, self.D):
            b_std = b_std * self.b_omega_std[d]
        b_std[torch.abs(b_std) < eps_tiny] = 0.0
        return b_std
    

    def forward(self, X):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.

        Returns:
            output: torch.tensor, [batch_size, output_dim], the output data.
        """
        # Retrieve the entire bias and weights standard deviation        
        W_std = self.get_W_std()
        b_std = self.get_b_std()
        W = self.W_mu + af.adapted_softplus(W_std) *\
            torch.randn((self.n_in, self.n_out), device=self.device)
        if self.scaled_variance:
            W = W / math.sqrt(self.n_in)
        b = self.b_mu + af.adapted_softplus(b_std) *\
            torch.randn((self.n_out), device=self.device)

        output = torch.mm(X, W) + b

        return output


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
        # Retrieve the entire bias and weights standard deviation
        W_std = self.get_W_std()
        b_std = self.get_b_std()
        Ws = self.W_mu + af.adapted_softplus(W_std) *\
            torch.randn([n_samples, self.n_in, self.n_out],
                        device=self.device)

        if self.scaled_variance:
            Ws = Ws / math.sqrt(self.n_in)
        bs = self.b_mu + af.adapted_softplus(b_std) *\
            torch.randn([n_samples, 1, self.n_out],
                        device=self.device)
        return torch.matmul(X, Ws) + bs

    def get_W_std_mask(self):
        W_std = self.get_W_std()
        return W_std == 0


    def get_b_std_mask(self):
        b_std = self.get_b_std()
        return b_std == 0