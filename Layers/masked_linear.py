import math
import torch
import torch.nn as nn
from Initialization.DWF_initlialization import dwf_initialization


class MaskedLinear(nn.Module):
    def __init__(self, n_in, n_out, b_det_mask, W_det_mask, D, W_std = None, b_std = None, scaled_variance=True, 
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
        self.D = D
        self.W_det_mask = W_det_mask.to(self.device)
        self.W_stoch_mask = ~W_det_mask.to(self.device)
        self.b_det_mask = b_det_mask.to(self.device)
        self.b_stoch_mask = ~b_det_mask.to(self.device)

        if W_std is None:
            if self.scaled_variance:
                W_std = 1.
            else:
                W_std = 1. / math.sqrt(self.n_in)
        if b_std is None:
            b_std = 1.

        # Register the masks and perform masking
        self.register_buffer("weight_mask", self.W_det_mask)  
        self.register_buffer("bias_mask", self.b_det_mask) 

        full_weight = torch.zeros(self.n_in, self.n_out, device = self.device)
        full_bias = torch.zeros(self.n_out, device = self.device)
        #self.W_det = nn.Parameter(full_weight[self.W_det_mask])
        self.W_stoch = nn.Parameter(full_weight[self.W_stoch_mask])
        #self.b_det = nn.Parameter(full_bias[self.b_det_mask])
        self.b_stoch = nn.Parameter(full_bias[self.b_stoch_mask])
        W_shape = full_weight[self.W_det_mask].shape[0]
        b_shape = full_bias[self.b_det_mask].shape[0]
        self.W_det_omega = nn.ParameterList()
        self.b_det_omega = nn.ParameterList()
        omega_factors_W_std = dwf_initialization(
            [W_shape], [W_std], self.D, W_std*0.25, device = self.device
        )[0]  # shape (layer_size, D)
        # Store D separate factor tensors
        for d in range(self.D):
            param = nn.Parameter(omega_factors_W_std[:, d].view(W_shape), requires_grad=True) 
            self.W_det_omega.append(param)
        
        omega_factors_b_std = dwf_initialization(
            [b_shape], [b_std], self.D, b_std*0.25, device = self.device
        )[0]  # shape (layer_size, D)
        # Store D separate factor tensors
        for d in range(self.D):
            param = nn.Parameter(omega_factors_b_std[:, d].view(b_shape), requires_grad=True) 
            self.b_det_omega.append(param)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # Reinitialize only the weights where the mask is 1
            std = 1.
            if not self.scaled_variance:
                std = std / math.sqrt(self.n_in)

        # Sample new weights/biases only for the stochastic parameters
        new_W = torch.rand(self.n_in, self.n_out, device=self.device)
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
        W_det = self.get_W_det()
        full_W = torch.zeros(self.n_in, self.n_out, device=self.device)
        full_W[self.W_det_mask] = W_det
        full_W[self.W_stoch_mask] = self.W_stoch
        return full_W
    
    def get_b(self):
        b_det = self.get_b_det()
        full_b = torch.zeros(self.n_out, device=self.device)
        full_b[self.b_det_mask] = b_det
        full_b[self.b_stoch_mask] = self.b_stoch
        return full_b
    
    def get_W_det(self, eps_tiny = 1.19e-7):
        """
        Multiplies all the weight matrices elementwise
        """
        W_det = self.W_det_omega[0]
        for d in range(1, self.D):
            W_det = W_det * self.W_det_omega[d]
        W_det[torch.abs(W_det) < eps_tiny] = 0.0
        return W_det

    def get_b_det(self, eps_tiny = 1.19e-7):
        """
        Multiplies all the bias vectors elementwise
        """
        b_det = self.b_det_omega[0]
        for d in range(1, self.D):
            b_det = b_det * self.b_det_omega[d]
        b_det[torch.abs(b_det) < eps_tiny] = 0.0
        return b_det