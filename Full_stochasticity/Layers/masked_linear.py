import math
import torch
import torch.nn as nn

"""
This is a linear layer
It has two additional parameters, namely the masks indicating wheter the bias or weight at a certain index is
stochastic or deterministic (1 means deterministic and 0 means stochastic)
"""
class MaskedLinear(nn.Module):
    def __init__(self, n_in, n_out, b_det_mask, W_det_mask, W_std = None, b_std = None, scaled_variance=True, 
                 device = "cpu"):
        """Initialization.

        Args:
            n_in: int, the size of the input data.
            n_out: int, the size of the output.
        """
        super(MaskedLinear, self).__init__()
        self.device = device
        self.n_in = n_in
        self.n_out = n_out
        self.scaled_variance = scaled_variance
        self.W_stoch_mask = ~W_det_mask.to(self.device)
        self.b_stoch_mask = ~b_det_mask.to(self.device)

        if W_std is None:
            if self.scaled_variance:
                W_std = 1.
            else:
                W_std = 1. / math.sqrt(self.n_in)
        if b_std is None:
            b_std = 1.

        # Register the masks and perform masking
        self.register_buffer("weight_mask", self.W_stoch_mask)  
        self.register_buffer("bias_mask", self.b_stoch_mask) 

        full_weight = torch.zeros(self.n_in, self.n_out, device = self.device)
        full_bias = torch.zeros(self.n_out, device = self.device)
        self.W_stoch = nn.Parameter(full_weight[self.W_stoch_mask])
        self.b_stoch = nn.Parameter(full_bias[self.b_stoch_mask])
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # Reinitialize only the weights where the mask is 1
            std = 1.
            if not self.scaled_variance:
                std = std / math.sqrt(self.n_in)

        # Sample new weights/biases only for the stochastic parameters
        new_W = torch.rand(self.n_in, self.n_out, device=self.device) * std
        new_b = torch.zeros(self.n_out, device = self.device)

        self.W_stoch.data = new_W[self.W_stoch_mask]
        self.b_stoch.data = new_b[self.b_stoch_mask]

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
        W = self.get_W()
        b = self.get_b()
        Ws = W.repeat(n_samples, 1, 1)
        bs = b.repeat(n_samples, 1, 1)

        return torch.matmul(X, Ws) + bs

    def get_W(self):
        """
        Returns a properly shaped W where the deterministic weights are simply set to 0
        """
        full_W = torch.zeros(self.n_in, self.n_out, device=self.device)
        full_W[self.W_stoch_mask] = self.W_stoch
        return full_W
    
    def get_b(self):
        """
        Retruns a properly shaped b where the deterministic parts of the bias are set to 0
        """
        full_b = torch.zeros(self.n_out, device=self.device)
        full_b[self.b_stoch_mask] = self.b_stoch
        return full_b

