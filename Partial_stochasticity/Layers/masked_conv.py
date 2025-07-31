import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Initialization.DWF_initlialization import dwf_initialization

"""
This defines a 2D Convolutional layer
The important thing is that here masks to signal which parameters are stochastic and which ones are 
deterministic are part of the arguments. True means deterministic and fals means stochastic.
Additionally, we perform DWF on the deterministic parameters
"""
class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, W_det_mask, b_det_mask = None,  D= 3, W_std = 1, b_std = 1, 
                 stride=1, padding=0, dilation=1, bias=True, scaled_variance=True, device = "cpu"):
        super(MaskedConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.scaled_variance = scaled_variance
        self.device = device
        self.D = D
        self.W_stoch_mask = ~W_det_mask.to(self.device)
        self.W_det_mask = W_det_mask.to(self.device)
        if b_det_mask:
            self.b_stoch_mask = ~b_det_mask.to(self.device)
            self.b_det_mask = b_det_mask.to(self.device)
        self.W_shape = (self.out_channels, self.in_channels, *self.kernel_size)
        self.b_shape = (self.out_channels)

        # Register the masks and perform masking
        self.register_buffer("weight_mask", self.W_stoch_mask)  
        if b_det_mask:
            self.register_buffer("bias_mask", self.b_stoch_mask) 

        full_weight = torch.zeros(self.W_shape, device=self.device)
        self.W_stoch = nn.Parameter(full_weight[self.W_stoch_mask])
        self.W_det_omega = nn.ParameterList()
        omega_factors_W = dwf_initialization(
            [self.W_det_mask.sum().item()], 
            [W_std], self.D, W_std*0.25, device = self.device)[0]  # shape (layer_size, D)
        # Store D separate factor tensors
        for d in range(self.D):
            param = nn.Parameter(omega_factors_W[:, d].view(self.W_det_mask.sum().item()), requires_grad=True) 
            self.W_det_omega.append(param)
        if self.bias:
            full_bias = torch.zeros(self.b_shape, device = self.device)
            self.b_stoch = nn.Parameter(full_bias[self.b_stoch_mask])
            self.b_det_omega = nn.ParameterList()
            omega_factors_b_std = dwf_initialization(
            [self.b_det_mask.sum().item()], [b_std], self.D, b_std*0.25, device = self.device)[0]  # shape (layer_size, D)
            # Store D separate factor tensors
            for d in range(self.D):
                param = nn.Parameter(omega_factors_b_std[:, d].view(self.b_det_mask.sum().item()), requires_grad=True) 
                self.b_det_omega.append(param)
        else:
            self.register_buffer("b", torch.zeros(self.b_shape))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.
        if not self.scaled_variance:
            std = std / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        new_W = torch.rand(self.W_shape, device = self.device) * std
        self.W_stoch.data = new_W[self.W_stoch_mask]
        if self.bias:
            new_b = torch.zeros(self.b_shape, device = self.device)
            self.b_stoch.data = new_b[self.b_stoch_mask]

    def forward(self, X):
        W = self.get_W()
        if self.scaled_variance:
            W = W / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        if self.bias:
            b = self.get_b()
        else:
            b = torch.zeros((self.out_channels), device=self.device)
        return F.conv2d(X, W, b, self.stride, self.padding, self.dilation)
    
    def get_W(self):
        W_det = self.get_W_det()
        full_W = torch.zeros(self.W_shape, device=self.device)
        full_W[self.W_det_mask] = W_det
        full_W[self.W_stoch_mask] = self.W_stoch
        return full_W
    
    def get_b(self):
        b_det = self.get_b_det()
        full_b = torch.zeros(self.b_shape, device=self.device)
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
    
    def get_nums_pruned_W(self):
        """
        Returns the number of pruned stochastic and deterministic weights
        """
        W = self.get_W()
        W_det = self.get_W_det()
        num_pruned_all = torch.sum(W == 0).item()
        num_pruned_det = torch.sum(W_det == 0).item()

        return num_pruned_all-num_pruned_det, num_pruned_det
    
    def get_nums_pruned_b(self):
        """
        Returns the number of pruned stochastic and deterministic biases
        """
        if self.bias:
            b = self.get_b()
            b_det = self.get_b_det()
            num_pruned_all = torch.sum(b == 0).item()
            num_pruned_det = torch.sum(b_det == 0).item()
        else:
            num_pruned_all = 0
            num_pruned_det = 0

        return num_pruned_all-num_pruned_det, num_pruned_det

